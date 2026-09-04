//! Bounded vector/text paint engine for the born-digital PDF subset admitted by
//! [`crate::pdf::PdfPages`].
//!
//! The supported dialect is deliberately concrete: deterministic score PDFs
//! emitted by MTDT plus the bounded simple-Type1C subset emitted by LilyPond.
//! It is still a real renderer. Paths are interpreted in content order,
//! embedded CIDFontType2 and Type1C programs are outlined through `ttf-parser`,
//! and a fixed supersampled rasterizer produces page pixels without FFI or a
//! helper executable. Unknown paint operators refuse rather than disappearing
//! from the page.

use std::collections::BTreeMap;

use image::{DynamicImage, GrayImage, Luma};
use lopdf::content::{Content, Operation};
use lopdf::{Dictionary, Document, Object, ObjectId};
use ttf_parser::{Face, GlyphId, OutlineBuilder, cff};

const SUPERSAMPLE: u32 = 4;
const MAX_VECTOR_OUTPUT_PIXELS: u64 = 1 << 24;
const MAX_VECTOR_WORKING_PIXELS: u64 = 1 << 28;
const MAX_VECTOR_PATH_SEGMENTS: usize = 2_000_000;
const MAX_VECTOR_CONTOURS: usize = 200_000;
const MAX_VECTOR_GLYPHS: usize = 1_000_000;
const MAX_VECTOR_FONT_BYTES: usize = 32 * 1024 * 1024;
const MAX_VECTOR_FONT_COUNT: usize = 32;
const MAX_TYPE1_GLYPH_NAME_BYTES: usize = 255;
const QUADRATIC_STEPS: usize = 12;
const CUBIC_STEPS: usize = 16;

#[derive(Clone, Copy, Debug)]
struct Matrix {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
}

impl Matrix {
    const IDENTITY: Self = Self {
        a: 1.0,
        b: 0.0,
        c: 0.0,
        d: 1.0,
        e: 0.0,
        f: 0.0,
    };

    fn concat(self, rhs: Self) -> Self {
        Self {
            a: self.a * rhs.a + self.c * rhs.b,
            b: self.b * rhs.a + self.d * rhs.b,
            c: self.a * rhs.c + self.c * rhs.d,
            d: self.b * rhs.c + self.d * rhs.d,
            e: self.a * rhs.e + self.c * rhs.f + self.e,
            f: self.b * rhs.e + self.d * rhs.f + self.f,
        }
    }

    fn point(self, point: Point) -> Point {
        Point {
            x: self.a * point.x + self.c * point.y + self.e,
            y: self.b * point.x + self.d * point.y + self.f,
        }
    }

    fn is_finite(self) -> bool {
        [self.a, self.b, self.c, self.d, self.e, self.f]
            .into_iter()
            .all(f64::is_finite)
    }

    fn similarity_scale(self) -> Result<f64, String> {
        let x = self.a.hypot(self.b);
        let y = self.c.hypot(self.d);
        let dot = self.a * self.c + self.b * self.d;
        let tolerance = x.max(y).max(1.0) * 1.0e-9;
        if x <= f64::EPSILON
            || y <= f64::EPSILON
            || (x - y).abs() > tolerance
            || dot.abs() > tolerance * x.max(y)
        {
            return Err(
                "stroke CTM has anisotropic scale or shear outside the admitted vector subset"
                    .to_owned(),
            );
        }
        Ok((x + y) * 0.5)
    }
}

#[derive(Clone, Copy, Debug)]
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn lerp(self, rhs: Self, t: f64) -> Self {
        Self {
            x: self.x + (rhs.x - self.x) * t,
            y: self.y + (rhs.y - self.y) * t,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct Contour {
    points: Vec<Point>,
    closed: bool,
}

#[derive(Clone, Debug, Default)]
struct PaintPath {
    contours: Vec<Contour>,
    segment_count: usize,
}

impl PaintPath {
    fn move_to(&mut self, point: Point) -> Result<(), String> {
        if self.contours.len() >= MAX_VECTOR_CONTOURS {
            return Err(format!(
                "vector page exceeds the {MAX_VECTOR_CONTOURS}-contour limit"
            ));
        }
        self.contours.push(Contour {
            points: vec![point],
            closed: false,
        });
        Ok(())
    }

    fn current_point(&self) -> Result<Point, String> {
        self.contours
            .last()
            .and_then(|contour| contour.points.last())
            .copied()
            .ok_or_else(|| "path operation has no current point".to_owned())
    }

    fn line_to(&mut self, point: Point) -> Result<(), String> {
        self.add_segment()?;
        self.contours
            .last_mut()
            .ok_or_else(|| "line operation has no current subpath".to_owned())?
            .points
            .push(point);
        Ok(())
    }

    fn cubic_to(&mut self, c1: Point, c2: Point, end: Point) -> Result<(), String> {
        let start = self.current_point()?;
        for step in 1..=CUBIC_STEPS {
            let t = step as f64 / CUBIC_STEPS as f64;
            let mt = 1.0 - t;
            let point = Point {
                x: mt * mt * mt * start.x
                    + 3.0 * mt * mt * t * c1.x
                    + 3.0 * mt * t * t * c2.x
                    + t * t * t * end.x,
                y: mt * mt * mt * start.y
                    + 3.0 * mt * mt * t * c1.y
                    + 3.0 * mt * t * t * c2.y
                    + t * t * t * end.y,
            };
            self.line_to(point)?;
        }
        Ok(())
    }

    fn quadratic_to(&mut self, control: Point, end: Point) -> Result<(), String> {
        let start = self.current_point()?;
        for step in 1..=QUADRATIC_STEPS {
            let t = step as f64 / QUADRATIC_STEPS as f64;
            let mt = 1.0 - t;
            let point = Point {
                x: mt * mt * start.x + 2.0 * mt * t * control.x + t * t * end.x,
                y: mt * mt * start.y + 2.0 * mt * t * control.y + t * t * end.y,
            };
            self.line_to(point)?;
        }
        Ok(())
    }

    fn close(&mut self) -> Result<(), String> {
        let already_closed = self
            .contours
            .last()
            .ok_or_else(|| "closepath has no current subpath".to_owned())?
            .closed;
        if already_closed {
            return Ok(());
        }
        self.add_segment()?;
        self.contours
            .last_mut()
            .ok_or_else(|| "closepath current subpath disappeared".to_owned())?
            .closed = true;
        Ok(())
    }

    fn add_segment(&mut self) -> Result<(), String> {
        self.segment_count = self
            .segment_count
            .checked_add(1)
            .ok_or_else(|| "vector path segment count overflow".to_owned())?;
        if self.segment_count > MAX_VECTOR_PATH_SEGMENTS {
            return Err(format!(
                "vector page exceeds the {MAX_VECTOR_PATH_SEGMENTS}-segment limit"
            ));
        }
        Ok(())
    }
}

#[derive(Default)]
struct RenderBudget {
    path_segments: usize,
    contours: usize,
    glyphs: usize,
}

impl RenderBudget {
    fn charge_path_growth(
        &mut self,
        path: &PaintPath,
        prior_segments: usize,
        prior_contours: usize,
    ) -> Result<(), String> {
        self.charge(
            path.segment_count.saturating_sub(prior_segments),
            path.contours.len().saturating_sub(prior_contours),
        )
    }

    fn charge_complete_path(&mut self, path: &PaintPath) -> Result<(), String> {
        self.charge(path.segment_count, path.contours.len())
    }

    fn charge(&mut self, segments: usize, contours: usize) -> Result<(), String> {
        self.path_segments = self
            .path_segments
            .checked_add(segments)
            .ok_or_else(|| "vector page segment budget overflow".to_owned())?;
        self.contours = self
            .contours
            .checked_add(contours)
            .ok_or_else(|| "vector page contour budget overflow".to_owned())?;
        if self.path_segments > MAX_VECTOR_PATH_SEGMENTS {
            return Err(format!(
                "vector page exceeds the cumulative {MAX_VECTOR_PATH_SEGMENTS}-segment limit"
            ));
        }
        if self.contours > MAX_VECTOR_CONTOURS {
            return Err(format!(
                "vector page exceeds the cumulative {MAX_VECTOR_CONTOURS}-contour limit"
            ));
        }
        Ok(())
    }

    fn charge_glyph(&mut self) -> Result<(), String> {
        self.glyphs = self
            .glyphs
            .checked_add(1)
            .ok_or_else(|| "vector page glyph budget overflow".to_owned())?;
        if self.glyphs > MAX_VECTOR_GLYPHS {
            return Err(format!(
                "vector page exceeds the {MAX_VECTOR_GLYPHS}-glyph limit"
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct GraphicsState {
    ctm: Matrix,
    line_width: f64,
    dash: Vec<f64>,
    dash_phase: f64,
    line_cap: LineCap,
    line_join: LineJoin,
}

impl Default for GraphicsState {
    fn default() -> Self {
        Self {
            ctm: Matrix::IDENTITY,
            line_width: 1.0,
            dash: Vec::new(),
            dash_phase: 0.0,
            line_cap: LineCap::Butt,
            line_join: LineJoin::Miter,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LineCap {
    Butt,
    Round,
    Square,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LineJoin {
    Miter,
    Round,
    Bevel,
}

#[derive(Clone, Debug)]
struct TextState {
    matrix: Matrix,
    line_matrix: Matrix,
    font_name: Option<Vec<u8>>,
    font_size: f64,
    leading: f64,
}

#[derive(Clone, Debug)]
struct PersistentTextState {
    font_name: Option<Vec<u8>>,
    font_size: f64,
    leading: f64,
}

impl Default for TextState {
    fn default() -> Self {
        Self {
            matrix: Matrix::IDENTITY,
            line_matrix: Matrix::IDENTITY,
            font_name: None,
            font_size: 0.0,
            leading: 0.0,
        }
    }
}

impl TextState {
    fn begin_object(&mut self) {
        self.matrix = Matrix::IDENTITY;
        self.line_matrix = Matrix::IDENTITY;
    }

    fn persistent(&self) -> PersistentTextState {
        PersistentTextState {
            font_name: self.font_name.clone(),
            font_size: self.font_size,
            leading: self.leading,
        }
    }

    fn restore_persistent(&mut self, saved: PersistentTextState) {
        self.font_name = saved.font_name;
        self.font_size = saved.font_size;
        self.leading = saved.leading;
    }
}

#[derive(Debug)]
pub(super) struct CidWidths {
    default: f64,
    explicit: BTreeMap<u16, f64>,
}

impl CidWidths {
    pub(super) fn get(&self, cid: u16) -> f64 {
        self.explicit.get(&cid).copied().unwrap_or(self.default)
    }
}

#[derive(Debug)]
enum EmbeddedFont {
    IdentityHTrueType {
        bytes: Vec<u8>,
        widths: CidWidths,
    },
    SimpleType1C {
        bytes: Vec<u8>,
        glyph_names: Vec<Option<String>>,
        widths: Vec<Option<f64>>,
    },
}

/// Validated embedded Type1C program and its effective one-byte PDF encoding.
///
/// Selectable-text extraction consumes this same provider-owned loader so it
/// cannot admit a simple font that the renderer would reject.
pub(super) struct SimpleType1CProgram {
    pub(super) bytes: Vec<u8>,
    pub(super) glyph_names: Vec<Option<String>>,
}

struct TextRenderContext<'a> {
    graphics: &'a GraphicsState,
    text: &'a mut TextState,
    raster: &'a mut SupersampledRaster,
    budget: &'a mut RenderBudget,
}

#[derive(Clone, Copy)]
enum FillRule {
    NonZero,
    EvenOdd,
}

struct SupersampledRaster {
    width: u32,
    height: u32,
    high_width: u32,
    high_height: u32,
    bounds: [f64; 4],
    ink: Vec<u8>,
}

impl SupersampledRaster {
    fn new(bounds: [f64; 4]) -> Result<Self, String> {
        if !bounds.into_iter().all(f64::is_finite)
            || bounds[2] <= bounds[0]
            || bounds[3] <= bounds[1]
        {
            return Err(format!("invalid vector page bounds {bounds:?}"));
        }
        let width = ceil_dimension(bounds[2] - bounds[0], "width")?;
        let height = ceil_dimension(bounds[3] - bounds[1], "height")?;
        let output_pixels = u64::from(width)
            .checked_mul(u64::from(height))
            .ok_or_else(|| "vector output pixel count overflow".to_owned())?;
        if output_pixels > MAX_VECTOR_OUTPUT_PIXELS {
            return Err(format!(
                "vector page has {output_pixels} output pixels, exceeding the {MAX_VECTOR_OUTPUT_PIXELS}-pixel limit"
            ));
        }
        let high_width = width
            .checked_mul(SUPERSAMPLE)
            .ok_or_else(|| "vector supersample width overflow".to_owned())?;
        let high_height = height
            .checked_mul(SUPERSAMPLE)
            .ok_or_else(|| "vector supersample height overflow".to_owned())?;
        let working_pixels = u64::from(high_width)
            .checked_mul(u64::from(high_height))
            .ok_or_else(|| "vector working pixel count overflow".to_owned())?;
        if working_pixels > MAX_VECTOR_WORKING_PIXELS {
            return Err(format!(
                "vector page needs {working_pixels} supersampled pixels, exceeding the {MAX_VECTOR_WORKING_PIXELS}-pixel limit"
            ));
        }
        let allocation = usize::try_from(working_pixels)
            .map_err(|_| "vector working buffer does not fit this host".to_owned())?;
        Ok(Self {
            width,
            height,
            high_width,
            high_height,
            bounds,
            ink: vec![0; allocation],
        })
    }

    fn device_point(&self, point: Point) -> Point {
        Point {
            x: (point.x - self.bounds[0]) * f64::from(SUPERSAMPLE),
            y: (self.bounds[3] - point.y) * f64::from(SUPERSAMPLE),
        }
    }

    fn fill(&mut self, path: &PaintPath, rule: FillRule) -> Result<(), String> {
        let contours = self.device_contours(path)?;
        let Some((first_row, last_row)) = contour_row_bounds(&contours, self.high_height) else {
            return Ok(());
        };
        for row in first_row..=last_row {
            let sample_y = f64::from(row) + 0.5;
            let mut crossings = Vec::<(f64, i32)>::new();
            for contour in &contours {
                if contour.len() < 2 {
                    continue;
                }
                for edge in contour.windows(2) {
                    add_crossing(&mut crossings, edge[0], edge[1], sample_y);
                }
                if let (Some(first), Some(last)) = (contour.first(), contour.last()) {
                    add_crossing(&mut crossings, *last, *first, sample_y);
                }
            }
            crossings.sort_by(|left, right| left.0.total_cmp(&right.0));
            match rule {
                FillRule::EvenOdd => self.fill_even_odd_row(row, &crossings),
                FillRule::NonZero => self.fill_nonzero_row(row, &crossings),
            }
        }
        Ok(())
    }

    fn stroke(
        &mut self,
        path: &PaintPath,
        width: f64,
        dash: &[f64],
        dash_phase: f64,
        line_cap: LineCap,
        line_join: LineJoin,
    ) -> Result<(), String> {
        if !width.is_finite() || width < 0.0 {
            return Err(format!("invalid vector stroke width {width}"));
        }
        if dash.iter().any(|value| !value.is_finite() || *value < 0.0) {
            return Err("invalid vector dash array".to_owned());
        }
        if !dash.is_empty() && dash.iter().all(|value| *value == 0.0) {
            return Err("vector dash array cannot contain only zero lengths".to_owned());
        }
        let effective_dash = !dash.is_empty();
        for contour in &path.contours {
            if contour.points.len() < 2 {
                continue;
            }
            let mut pattern = effective_dash
                .then(|| DashCursor::new(dash, dash_phase))
                .transpose()?;
            let edge_count = contour.points.len() - 1;
            for (index, edge) in contour.points.windows(2).enumerate() {
                let caps = if pattern.is_some() {
                    (line_cap, line_cap)
                } else {
                    (
                        if index == 0 && !contour.closed {
                            line_cap
                        } else {
                            LineCap::Butt
                        },
                        if index + 1 == edge_count && !contour.closed {
                            line_cap
                        } else {
                            LineCap::Butt
                        },
                    )
                };
                self.stroke_edge(edge[0], edge[1], width, pattern.as_mut(), caps)?;
            }
            if contour.closed
                && let (Some(first), Some(last)) = (contour.points.first(), contour.points.last())
            {
                let caps = if pattern.is_some() {
                    (line_cap, line_cap)
                } else {
                    (LineCap::Butt, LineCap::Butt)
                };
                self.stroke_edge(*last, *first, width, pattern.as_mut(), caps)?;
            }
            if pattern.is_none() {
                for points in contour.points.windows(3) {
                    self.draw_join(points[0], points[1], points[2], width, line_join)?;
                }
                if contour.closed && contour.points.len() >= 3 {
                    let last = contour.points.len() - 1;
                    self.draw_join(
                        contour.points[last - 1],
                        contour.points[last],
                        contour.points[0],
                        width,
                        line_join,
                    )?;
                    self.draw_join(
                        contour.points[last],
                        contour.points[0],
                        contour.points[1],
                        width,
                        line_join,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn stroke_edge(
        &mut self,
        start: Point,
        end: Point,
        width: f64,
        mut dash: Option<&mut DashCursor<'_>>,
        caps: (LineCap, LineCap),
    ) -> Result<(), String> {
        let length = (end.x - start.x).hypot(end.y - start.y);
        if length <= f64::EPSILON {
            return Ok(());
        }
        if let Some(cursor) = dash.as_mut() {
            let mut consumed = 0.0;
            while consumed < length {
                let take = cursor.remaining.min(length - consumed);
                if take <= f64::EPSILON {
                    cursor.advance_slot();
                    continue;
                }
                if cursor.ink {
                    let left = start.lerp(end, consumed / length);
                    let right = start.lerp(end, (consumed + take) / length);
                    self.draw_solid_segment(left, right, width, caps)?;
                }
                consumed += take;
                cursor.remaining -= take;
                if cursor.remaining <= f64::EPSILON {
                    cursor.advance_slot();
                }
            }
        } else {
            self.draw_solid_segment(start, end, width, caps)?;
        }
        Ok(())
    }

    fn draw_solid_segment(
        &mut self,
        start: Point,
        end: Point,
        width: f64,
        caps: (LineCap, LineCap),
    ) -> Result<(), String> {
        let mut start = self.device_point(start);
        let mut end = self.device_point(end);
        // A legal subpixel hairline must still contribute coverage. At the
        // fixed 4x grid, a radius below half a sample can fall exactly between
        // sample centers and disappear; the half-sample floor is the bounded
        // coverage analogue of PDF hairline painting.
        let radius = if width == 0.0 {
            f64::from(SUPERSAMPLE) * 0.5
        } else {
            (width * f64::from(SUPERSAMPLE) * 0.5).max(0.5)
        };
        let segment_dx = end.x - start.x;
        let segment_dy = end.y - start.y;
        let segment_length = segment_dx.hypot(segment_dy);
        if segment_length <= f64::EPSILON {
            return Ok(());
        }
        let unit_x = segment_dx / segment_length;
        let unit_y = segment_dy / segment_length;
        if caps.0 == LineCap::Square {
            start.x -= unit_x * radius;
            start.y -= unit_y * radius;
        }
        if caps.1 == LineCap::Square {
            end.x += unit_x * radius;
            end.y += unit_y * radius;
        }
        let min_x = ((start.x.min(end.x) - radius).floor() as i64).max(0);
        let max_x = ((start.x.max(end.x) + radius).ceil() as i64)
            .min(i64::from(self.high_width).saturating_sub(1));
        let min_y = ((start.y.min(end.y) - radius).floor() as i64).max(0);
        let max_y = ((start.y.max(end.y) + radius).ceil() as i64)
            .min(i64::from(self.high_height).saturating_sub(1));
        if min_x > max_x || min_y > max_y {
            return Ok(());
        }
        let dx = end.x - start.x;
        let dy = end.y - start.y;
        let denom = dx * dx + dy * dy;
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let px = x as f64 + 0.5;
                let py = y as f64 + 0.5;
                let raw_t = if denom <= f64::EPSILON {
                    0.0
                } else {
                    ((px - start.x) * dx + (py - start.y) * dy) / denom
                };
                if (raw_t < 0.0 && caps.0 != LineCap::Round)
                    || (raw_t > 1.0 && caps.1 != LineCap::Round)
                {
                    continue;
                }
                let t = raw_t.clamp(0.0, 1.0);
                let nearest_x = start.x + t * dx;
                let nearest_y = start.y + t * dy;
                if (px - nearest_x).hypot(py - nearest_y) <= radius {
                    self.mark(x as u32, y as u32)?;
                }
            }
        }
        Ok(())
    }

    fn draw_join(
        &mut self,
        previous: Point,
        vertex: Point,
        next: Point,
        width: f64,
        join: LineJoin,
    ) -> Result<(), String> {
        let previous_dx = vertex.x - previous.x;
        let previous_dy = vertex.y - previous.y;
        let next_dx = next.x - vertex.x;
        let next_dy = next.y - vertex.y;
        let previous_length = previous_dx.hypot(previous_dy);
        let next_length = next_dx.hypot(next_dy);
        if previous_length <= f64::EPSILON || next_length <= f64::EPSILON {
            return Ok(());
        }
        let previous_direction = Point {
            x: previous_dx / previous_length,
            y: previous_dy / previous_length,
        };
        let next_direction = Point {
            x: next_dx / next_length,
            y: next_dy / next_length,
        };
        let cross =
            previous_direction.x * next_direction.y - previous_direction.y * next_direction.x;
        if cross.abs() <= f64::EPSILON {
            return Ok(());
        }
        let radius = width * 0.5;
        if join == LineJoin::Round {
            return self.draw_page_disk(vertex, radius);
        }
        let side = if cross > 0.0 { -1.0 } else { 1.0 };
        let previous_outer = Point {
            x: vertex.x - previous_direction.y * radius * side,
            y: vertex.y + previous_direction.x * radius * side,
        };
        let next_outer = Point {
            x: vertex.x - next_direction.y * radius * side,
            y: vertex.y + next_direction.x * radius * side,
        };
        let mut contour = vec![vertex, previous_outer];
        if join == LineJoin::Miter {
            let delta = Point {
                x: next_outer.x - previous_outer.x,
                y: next_outer.y - previous_outer.y,
            };
            let t = (delta.x * next_direction.y - delta.y * next_direction.x) / cross;
            let intersection = Point {
                x: previous_outer.x + previous_direction.x * t,
                y: previous_outer.y + previous_direction.y * t,
            };
            if (intersection.x - vertex.x).hypot(intersection.y - vertex.y) <= radius * 10.0 {
                contour.push(intersection);
            }
        }
        contour.push(next_outer);
        self.fill(
            &PaintPath {
                contours: vec![Contour {
                    points: contour,
                    closed: true,
                }],
                segment_count: 0,
            },
            FillRule::NonZero,
        )
    }

    fn draw_page_disk(&mut self, center: Point, radius: f64) -> Result<(), String> {
        let center = self.device_point(center);
        let radius = (radius * f64::from(SUPERSAMPLE)).max(0.5);
        let min_x = ((center.x - radius).floor() as i64).max(0);
        let max_x =
            ((center.x + radius).ceil() as i64).min(i64::from(self.high_width).saturating_sub(1));
        let min_y = ((center.y - radius).floor() as i64).max(0);
        let max_y =
            ((center.y + radius).ceil() as i64).min(i64::from(self.high_height).saturating_sub(1));
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                let dx = x as f64 + 0.5 - center.x;
                let dy = y as f64 + 0.5 - center.y;
                if dx.hypot(dy) <= radius {
                    self.mark(x as u32, y as u32)?;
                }
            }
        }
        Ok(())
    }

    fn device_contours(&self, path: &PaintPath) -> Result<Vec<Vec<Point>>, String> {
        path.contours
            .iter()
            .map(|contour| {
                contour
                    .points
                    .iter()
                    .map(|point| {
                        let device = self.device_point(*point);
                        if !device.x.is_finite() || !device.y.is_finite() {
                            return Err("vector path transformed to a non-finite point".to_owned());
                        }
                        Ok(device)
                    })
                    .collect()
            })
            .collect()
    }

    fn fill_even_odd_row(&mut self, row: u32, crossings: &[(f64, i32)]) {
        let (pairs, _) = crossings.as_chunks::<2>();
        for pair in pairs {
            self.fill_span(row, pair[0].0, pair[1].0);
        }
    }

    fn fill_nonzero_row(&mut self, row: u32, crossings: &[(f64, i32)]) {
        let mut winding = 0i32;
        let mut start = None;
        for (x, delta) in crossings {
            let was_inside = winding != 0;
            winding += *delta;
            let is_inside = winding != 0;
            if !was_inside && is_inside {
                start = Some(*x);
            } else if was_inside
                && !is_inside
                && let Some(left) = start.take()
            {
                self.fill_span(row, left, *x);
            }
        }
    }

    fn fill_span(&mut self, row: u32, left: f64, right: f64) {
        if !left.is_finite() || !right.is_finite() || right <= left {
            return;
        }
        let first = ((left - 0.5).ceil() as i64).max(0);
        let last =
            ((right - 0.5).ceil() as i64 - 1).min(i64::from(self.high_width).saturating_sub(1));
        if first > last {
            return;
        }
        for x in first..=last {
            let _ = self.mark(x as u32, row);
        }
    }

    fn mark(&mut self, x: u32, y: u32) -> Result<(), String> {
        if x >= self.high_width || y >= self.high_height {
            return Ok(());
        }
        let index = u64::from(y)
            .checked_mul(u64::from(self.high_width))
            .and_then(|base| base.checked_add(u64::from(x)))
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| "vector working-buffer index overflow".to_owned())?;
        if let Some(pixel) = self.ink.get_mut(index) {
            *pixel = 1;
        }
        Ok(())
    }

    fn finish(self) -> Result<DynamicImage, String> {
        let mut output = GrayImage::from_pixel(self.width, self.height, Luma([255]));
        let samples = SUPERSAMPLE * SUPERSAMPLE;
        for y in 0..self.height {
            for x in 0..self.width {
                let mut covered = 0u32;
                for sy in 0..SUPERSAMPLE {
                    for sx in 0..SUPERSAMPLE {
                        let hx = x * SUPERSAMPLE + sx;
                        let hy = y * SUPERSAMPLE + sy;
                        let index = u64::from(hy)
                            .checked_mul(u64::from(self.high_width))
                            .and_then(|base| base.checked_add(u64::from(hx)))
                            .and_then(|value| usize::try_from(value).ok())
                            .ok_or_else(|| "vector downsample index overflow".to_owned())?;
                        covered += u32::from(self.ink.get(index).copied().unwrap_or(0));
                    }
                }
                let gray = 255u32.saturating_sub((covered * 255 + samples / 2) / samples) as u8;
                output.put_pixel(x, y, Luma([gray]));
            }
        }
        Ok(DynamicImage::ImageLuma8(output))
    }
}

struct DashCursor<'a> {
    pattern: &'a [f64],
    index: usize,
    remaining: f64,
    ink: bool,
}

impl<'a> DashCursor<'a> {
    fn new(pattern: &'a [f64], phase: f64) -> Result<Self, String> {
        if pattern.is_empty() {
            return Err("internal empty dash cursor".to_owned());
        }
        let period: f64 = pattern.iter().sum();
        if !period.is_finite() || period <= 0.0 || !phase.is_finite() {
            return Err("invalid vector dash period or phase".to_owned());
        }
        let mut cursor = Self {
            pattern,
            index: 0,
            remaining: pattern[0],
            ink: true,
        };
        let mut skip = phase.rem_euclid(period);
        while skip > 0.0 {
            if cursor.remaining <= f64::EPSILON {
                cursor.advance_slot();
                continue;
            }
            let take = skip.min(cursor.remaining);
            skip -= take;
            cursor.remaining -= take;
            if cursor.remaining <= f64::EPSILON {
                cursor.advance_slot();
            }
        }
        Ok(cursor)
    }

    fn advance_slot(&mut self) {
        self.index = (self.index + 1) % self.pattern.len();
        self.remaining = self.pattern[self.index];
        self.ink = self.index.is_multiple_of(2);
    }
}

fn ceil_dimension(value: f64, name: &str) -> Result<u32, String> {
    if !value.is_finite() || value <= 0.0 || value > f64::from(u32::MAX) {
        return Err(format!("invalid vector page {name} {value}"));
    }
    Ok(value.ceil() as u32)
}

fn add_crossing(crossings: &mut Vec<(f64, i32)>, start: Point, end: Point, y: f64) {
    if (start.y <= y && end.y > y) || (end.y <= y && start.y > y) {
        let t = (y - start.y) / (end.y - start.y);
        let x = start.x + t * (end.x - start.x);
        crossings.push((x, if end.y > start.y { 1 } else { -1 }));
    }
}

fn contour_row_bounds(contours: &[Vec<Point>], height: u32) -> Option<(u32, u32)> {
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    for point in contours.iter().flatten() {
        minimum = minimum.min(point.y);
        maximum = maximum.max(point.y);
    }
    if !minimum.is_finite() || !maximum.is_finite() || maximum < 0.0 || minimum >= f64::from(height)
    {
        return None;
    }
    let first = (minimum.floor() as i64).max(0) as u32;
    let last = (maximum.ceil() as i64)
        .min(i64::from(height).saturating_sub(1))
        .max(0) as u32;
    (first <= last).then_some((first, last))
}

struct GlyphOutline<'a> {
    path: &'a mut PaintPath,
    matrix: Matrix,
    failure: Option<String>,
}

impl GlyphOutline<'_> {
    fn map(&self, x: f32, y: f32) -> Point {
        self.matrix.point(Point {
            x: f64::from(x),
            y: f64::from(y),
        })
    }

    fn record(&mut self, result: Result<(), String>) {
        if self.failure.is_none()
            && let Err(error) = result
        {
            self.failure = Some(error);
        }
    }
}

impl OutlineBuilder for GlyphOutline<'_> {
    fn move_to(&mut self, x: f32, y: f32) {
        let point = self.map(x, y);
        let result = self.path.move_to(point);
        self.record(result);
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let point = self.map(x, y);
        let result = self.path.line_to(point);
        self.record(result);
    }

    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let control = self.map(x1, y1);
        let end = self.map(x, y);
        let result = self.path.quadratic_to(control, end);
        self.record(result);
    }

    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let control_1 = self.map(x1, y1);
        let control_2 = self.map(x2, y2);
        let end = self.map(x, y);
        let result = self.path.cubic_to(control_1, control_2, end);
        self.record(result);
    }

    fn close(&mut self) {
        let result = self.path.close();
        self.record(result);
    }
}

/// Render the exact vector/text content dialect emitted by MTDT's score-PDF
/// writer. The caller owns bounded content decoding and page rotation.
pub(crate) fn render_mtdt_vector_page(
    doc: &Document,
    page_id: ObjectId,
    content: &Content<Vec<Operation>>,
    bounds: [f64; 4],
) -> Result<DynamicImage, String> {
    let mut raster = SupersampledRaster::new(bounds)?;
    let mut graphics = GraphicsState::default();
    let mut stack = Vec::<(GraphicsState, PersistentTextState)>::new();
    let mut path = PaintPath::default();
    let mut text = TextState::default();
    let mut in_text = false;
    let mut marked_content_depth = 0usize;
    let mut fonts = BTreeMap::<Vec<u8>, EmbeddedFont>::new();
    let mut budget = RenderBudget::default();

    for (operation_index, operation) in content.operations.iter().enumerate() {
        validate_operator_context(&operation.operator, in_text).map_err(|error| {
            format!(
                "vector content operation {} ({}) failed: {error}",
                operation_index, operation.operator
            )
        })?;
        let prior_segments = path.segment_count;
        let prior_contours = path.contours.len();
        let result = (|| -> Result<(), String> {
            match operation.operator.as_str() {
                "q" => {
                    require_no_operands(&operation.operands, "q")?;
                    if stack.len() >= 64 {
                        Err("vector graphics-state stack exceeds 64 entries".to_owned())
                    } else {
                        stack.push((graphics.clone(), text.persistent()));
                        Ok(())
                    }
                }
                "Q" => {
                    require_no_operands(&operation.operands, "Q")?;
                    let (saved_graphics, saved_text) = stack
                        .pop()
                        .ok_or_else(|| "unbalanced vector graphics-state restore (Q)".to_owned())?;
                    graphics = saved_graphics;
                    text.restore_persistent(saved_text);
                    Ok(())
                }
                "cm" => {
                    let matrix = matrix_operands(&operation.operands)?;
                    graphics.ctm = graphics.ctm.concat(matrix);
                    if !graphics.ctm.is_finite() {
                        Err("vector CTM became non-finite".to_owned())
                    } else {
                        Ok(())
                    }
                }
                "w" => {
                    graphics.line_width = one_number(&operation.operands, "w")?;
                    if graphics.line_width < 0.0 {
                        Err(format!("invalid vector line width {}", graphics.line_width))
                    } else {
                        Ok(())
                    }
                }
                "J" => {
                    graphics.line_cap = line_cap_operand(&operation.operands)?;
                    Ok(())
                }
                "j" => {
                    graphics.line_join = line_join_operand(&operation.operands)?;
                    Ok(())
                }
                "d" => {
                    let (dash, phase) = dash_operands(&operation.operands)?;
                    graphics.dash = dash;
                    graphics.dash_phase = phase;
                    Ok(())
                }
                "m" => path.move_to(
                    graphics
                        .ctm
                        .point(point_operands(&operation.operands, "m")?),
                ),
                "l" => path.line_to(
                    graphics
                        .ctm
                        .point(point_operands(&operation.operands, "l")?),
                ),
                "c" => {
                    let points = cubic_operands(&operation.operands)?;
                    path.cubic_to(
                        graphics.ctm.point(points[0]),
                        graphics.ctm.point(points[1]),
                        graphics.ctm.point(points[2]),
                    )
                }
                "re" => append_rectangle(&mut path, graphics.ctm, &operation.operands),
                "h" => {
                    require_no_operands(&operation.operands, "h")?;
                    path.close()
                }
                "S" => {
                    require_no_operands(&operation.operands, "S")?;
                    stroke_current_path(&mut raster, &path, &graphics)?;
                    path = PaintPath::default();
                    Ok(())
                }
                "s" => {
                    require_no_operands(&operation.operands, "s")?;
                    path.close()?;
                    stroke_current_path(&mut raster, &path, &graphics)?;
                    path = PaintPath::default();
                    Ok(())
                }
                "f" | "F" => {
                    require_no_operands(&operation.operands, &operation.operator)?;
                    raster.fill(&path, FillRule::NonZero)?;
                    path = PaintPath::default();
                    Ok(())
                }
                "f*" => {
                    require_no_operands(&operation.operands, "f*")?;
                    raster.fill(&path, FillRule::EvenOdd)?;
                    path = PaintPath::default();
                    Ok(())
                }
                "n" => {
                    require_no_operands(&operation.operands, "n")?;
                    path = PaintPath::default();
                    Ok(())
                }
                "BT" => {
                    require_no_operands(&operation.operands, "BT")?;
                    in_text = true;
                    text.begin_object();
                    Ok(())
                }
                "ET" => {
                    require_no_operands(&operation.operands, "ET")?;
                    if marked_content_depth != 0 {
                        return Err(
                            "ET closes a text object with unterminated marked content".to_owned()
                        );
                    }
                    in_text = false;
                    Ok(())
                }
                "BMC" => {
                    validate_marked_content_begin(&operation.operands, "BMC")?;
                    if marked_content_depth >= 64 {
                        Err("marked-content nesting exceeds 64 entries".to_owned())
                    } else {
                        marked_content_depth += 1;
                        Ok(())
                    }
                }
                "BDC" => {
                    validate_marked_content_begin(&operation.operands, "BDC")?;
                    if marked_content_depth >= 64 {
                        Err("marked-content nesting exceeds 64 entries".to_owned())
                    } else {
                        marked_content_depth += 1;
                        Ok(())
                    }
                }
                "EMC" => {
                    require_no_operands(&operation.operands, "EMC")?;
                    marked_content_depth = marked_content_depth
                        .checked_sub(1)
                        .ok_or_else(|| "EMC has no matching BMC/BDC".to_owned())?;
                    Ok(())
                }
                "Tf" => set_text_font(&mut text, &operation.operands),
                "Tm" => {
                    let matrix = matrix_operands(&operation.operands)?;
                    text.matrix = matrix;
                    text.line_matrix = matrix;
                    Ok(())
                }
                "Td" => translate_text_matrix(&mut text, &operation.operands),
                "TL" => set_text_leading(&mut text, &operation.operands),
                "TJ" | "Tj" | "'" => {
                    let mut context = TextRenderContext {
                        graphics: &graphics,
                        text: &mut text,
                        raster: &mut raster,
                        budget: &mut budget,
                    };
                    match operation.operator.as_str() {
                        "TJ" => {
                            render_tj(doc, page_id, &operation.operands, &mut fonts, &mut context)
                        }
                        "Tj" => render_tj_string(
                            doc,
                            page_id,
                            &operation.operands,
                            &mut fonts,
                            &mut context,
                        ),
                        "'" => render_next_line_string(
                            doc,
                            page_id,
                            &operation.operands,
                            &mut fonts,
                            &mut context,
                        ),
                        _ => unreachable!("outer match restricts text-show operators"),
                    }
                }
                "gs" => apply_lilypond_ext_gstate(doc, page_id, &operation.operands),
                // MTDT currently emits black-only score pages. Accept explicit black
                // declarations, but refuse invisible color changes rather than
                // pretending they were painted.
                "g" | "G" => require_black_gray(&operation.operands),
                "rg" | "RG" => require_black_rgb(&operation.operands),
                other => Err(format!(
                    "PDF operator {other:?} is outside the admitted MTDT vector/text subset"
                )),
            }
        })();
        let result =
            result.and_then(|()| budget.charge_path_growth(&path, prior_segments, prior_contours));
        result.map_err(|error| {
            format!(
                "vector content operation {} ({}) failed: {error}",
                operation_index, operation.operator
            )
        })?;
    }
    if !stack.is_empty() {
        return Err("unbalanced vector graphics-state save (q)".to_owned());
    }
    if in_text {
        return Err("unterminated vector text object (BT without ET)".to_owned());
    }
    if marked_content_depth != 0 {
        return Err("unterminated marked content (BMC/BDC without EMC)".to_owned());
    }
    raster.finish()
}

fn validate_operator_context(operator: &str, in_text: bool) -> Result<(), String> {
    let text_body_operator = matches!(
        operator,
        "Tf" | "Tm" | "Td" | "TL" | "TJ" | "Tj" | "'" | "BMC" | "BDC" | "EMC"
    );
    if operator == "BT" && in_text {
        return Err("nested BT in vector page".to_owned());
    }
    if operator == "ET" && !in_text {
        return Err("ET outside a text object".to_owned());
    }
    if text_body_operator && !in_text {
        return Err(format!("text operator {operator} appears outside BT/ET"));
    }
    if in_text && operator != "ET" && !text_body_operator {
        return Err(format!(
            "operator {operator} is not admitted inside a vector text object"
        ));
    }
    Ok(())
}

fn validate_marked_content_begin(operands: &[Object], operator: &str) -> Result<(), String> {
    match (operator, operands) {
        ("BMC", [Object::Name(_)]) => Ok(()),
        ("BDC", [Object::Name(_), Object::Dictionary(_)]) => Ok(()),
        ("BMC", _) => Err("BMC expects one name operand".to_owned()),
        ("BDC", _) => Err("BDC expects a name and direct properties dictionary".to_owned()),
        _ => Err(format!(
            "unsupported marked-content begin operator {operator}"
        )),
    }
}

fn stroke_current_path(
    raster: &mut SupersampledRaster,
    path: &PaintPath,
    graphics: &GraphicsState,
) -> Result<(), String> {
    let (width, dash, dash_phase) = page_space_stroke_parameters(graphics)?;
    raster.stroke(
        path,
        width,
        &dash,
        dash_phase,
        graphics.line_cap,
        graphics.line_join,
    )
}

fn page_space_stroke_parameters(graphics: &GraphicsState) -> Result<(f64, Vec<f64>, f64), String> {
    let scale = graphics.ctm.similarity_scale()?;
    let width = if graphics.line_width == 0.0 {
        0.0
    } else {
        graphics.line_width * scale
    };
    let dash = graphics
        .dash
        .iter()
        .map(|value| value * scale)
        .collect::<Vec<_>>();
    Ok((width, dash, graphics.dash_phase * scale))
}

fn render_tj(
    doc: &Document,
    page_id: ObjectId,
    operands: &[Object],
    fonts: &mut BTreeMap<Vec<u8>, EmbeddedFont>,
    context: &mut TextRenderContext<'_>,
) -> Result<(), String> {
    let [Object::Array(items)] = operands else {
        return Err("TJ expects one array operand".to_owned());
    };
    for item in items {
        match item {
            Object::String(bytes, _) => {
                render_embedded_font_bytes(doc, page_id, bytes, fonts, context)?;
            }
            Object::Integer(_) | Object::Real(_) => {
                let adjustment = number(item)?;
                advance_text(context.text, -adjustment * context.text.font_size / 1000.0)?;
            }
            _ => return Err("TJ array contains neither a string nor a number".to_owned()),
        }
    }
    Ok(())
}

fn render_tj_string(
    doc: &Document,
    page_id: ObjectId,
    operands: &[Object],
    fonts: &mut BTreeMap<Vec<u8>, EmbeddedFont>,
    context: &mut TextRenderContext<'_>,
) -> Result<(), String> {
    let [Object::String(bytes, _)] = operands else {
        return Err("Tj expects one string operand".to_owned());
    };
    render_embedded_font_bytes(doc, page_id, bytes, fonts, context)
}

fn render_next_line_string(
    doc: &Document,
    page_id: ObjectId,
    operands: &[Object],
    fonts: &mut BTreeMap<Vec<u8>, EmbeddedFont>,
    context: &mut TextRenderContext<'_>,
) -> Result<(), String> {
    next_text_line(context.text)?;
    render_tj_string(doc, page_id, operands, fonts, context)
}

fn render_embedded_font_bytes(
    doc: &Document,
    page_id: ObjectId,
    bytes: &[u8],
    fonts: &mut BTreeMap<Vec<u8>, EmbeddedFont>,
    context: &mut TextRenderContext<'_>,
) -> Result<(), String> {
    let font_name = context
        .text
        .font_name
        .as_ref()
        .ok_or_else(|| "text paint has no selected font".to_owned())?
        .clone();
    if !fonts.contains_key(&font_name) {
        if fonts.len() >= MAX_VECTOR_FONT_COUNT {
            return Err(format!(
                "vector page exceeds the {MAX_VECTOR_FONT_COUNT}-font limit"
            ));
        }
        let font = load_embedded_font(doc, page_id, &font_name)?;
        fonts.insert(font_name.clone(), font);
    }
    let font = fonts
        .get(&font_name)
        .ok_or_else(|| "selected vector font disappeared from cache".to_owned())?;
    match font {
        EmbeddedFont::IdentityHTrueType {
            bytes: font_bytes,
            widths,
        } => render_identity_h_true_type(bytes, font_bytes, widths, context),
        EmbeddedFont::SimpleType1C {
            bytes: font_bytes,
            glyph_names,
            widths,
        } => render_simple_type1c(bytes, font_bytes, glyph_names, widths, context),
    }
}

fn render_identity_h_true_type(
    bytes: &[u8],
    font_bytes: &[u8],
    widths: &CidWidths,
    context: &mut TextRenderContext<'_>,
) -> Result<(), String> {
    let (codes, remainder) = bytes.as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(format!(
            "Identity-H text string has odd byte length {}",
            bytes.len()
        ));
    }
    let face = Face::parse(font_bytes, 0)
        .map_err(|error| format!("parse embedded TrueType font: {error:?}"))?;
    let units_per_em = f64::from(face.units_per_em());
    if units_per_em <= 0.0 || !context.text.font_size.is_finite() || context.text.font_size <= 0.0 {
        return Err("invalid embedded-font units or text size".to_owned());
    }
    for code in codes {
        context.budget.charge_glyph()?;
        let cid = u16::from_be_bytes([code[0], code[1]]);
        let glyph_id = GlyphId(cid);
        let scale = context.text.font_size / units_per_em;
        let glyph_to_page = context
            .graphics
            .ctm
            .concat(context.text.matrix)
            .concat(Matrix {
                a: scale,
                b: 0.0,
                c: 0.0,
                d: scale,
                e: 0.0,
                f: 0.0,
            });
        let mut glyph_path = PaintPath::default();
        let mut builder = GlyphOutline {
            path: &mut glyph_path,
            matrix: glyph_to_page,
            failure: None,
        };
        let outlined = face.outline_glyph(glyph_id, &mut builder).is_some();
        if let Some(error) = builder.failure {
            return Err(format!(
                "glyph {} outline exceeds renderer bounds: {error}",
                glyph_id.0
            ));
        }
        context.budget.charge_complete_path(&glyph_path)?;
        if outlined {
            context.raster.fill(&glyph_path, FillRule::NonZero)?;
        }
        advance_text(
            context.text,
            widths.get(cid) * context.text.font_size / 1000.0,
        )?;
    }
    Ok(())
}

fn render_simple_type1c(
    bytes: &[u8],
    font_bytes: &[u8],
    glyph_names: &[Option<String>],
    widths: &[Option<f64>],
    context: &mut TextRenderContext<'_>,
) -> Result<(), String> {
    let table = cff::Table::parse(font_bytes)
        .ok_or_else(|| "embedded Type1C font is not a valid bounded CFF table".to_owned())?;
    let matrix = table.matrix();
    let values = [
        matrix.sx, matrix.ky, matrix.kx, matrix.sy, matrix.tx, matrix.ty,
    ];
    if values
        .iter()
        .any(|value| !value.is_finite() || value.abs() > 1_000_000.0)
    {
        return Err("embedded Type1C FontMatrix is non-finite or excessive".to_owned());
    }
    let glyph_to_text = Matrix {
        a: context.text.font_size * f64::from(matrix.sx),
        b: context.text.font_size * f64::from(matrix.ky),
        c: context.text.font_size * f64::from(matrix.kx),
        d: context.text.font_size * f64::from(matrix.sy),
        e: context.text.font_size * f64::from(matrix.tx),
        f: context.text.font_size * f64::from(matrix.ty),
    };
    if !glyph_to_text.is_finite() {
        return Err("embedded Type1C glyph matrix is non-finite".to_owned());
    }
    for code in bytes {
        context.budget.charge_glyph()?;
        let index = usize::from(*code);
        let glyph_name = glyph_names
            .get(index)
            .and_then(Option::as_deref)
            .ok_or_else(|| format!("Type1 WinAnsi code {code} has no admitted glyph name"))?;
        let glyph_id = table.glyph_index_by_name(glyph_name).ok_or_else(|| {
            format!("Type1 WinAnsi code {code} names missing CFF glyph /{glyph_name}")
        })?;
        let glyph_to_page = context
            .graphics
            .ctm
            .concat(context.text.matrix)
            .concat(glyph_to_text);
        let mut glyph_path = PaintPath::default();
        let mut builder = GlyphOutline {
            path: &mut glyph_path,
            matrix: glyph_to_page,
            failure: None,
        };
        let outline = table.outline(glyph_id, &mut builder);
        if let Some(error) = builder.failure {
            return Err(format!(
                "Type1C glyph /{glyph_name} outline exceeds renderer bounds: {error}"
            ));
        }
        context.budget.charge_complete_path(&glyph_path)?;
        match outline {
            Ok(_) => context.raster.fill(&glyph_path, FillRule::NonZero)?,
            Err(ttf_parser::CFFError::ZeroBBox) => {}
            Err(error) => {
                return Err(format!(
                    "outline Type1C glyph /{glyph_name} for code {code}: {error:?}"
                ));
            }
        }
        let width = widths
            .get(index)
            .and_then(|width| *width)
            .ok_or_else(|| format!("Type1 WinAnsi code {code} has no declared PDF width"))?;
        advance_text(context.text, width * context.text.font_size / 1000.0)?;
    }
    Ok(())
}

fn advance_text(text: &mut TextState, amount: f64) -> Result<(), String> {
    text.matrix.e += text.matrix.a * amount;
    text.matrix.f += text.matrix.b * amount;
    if text.matrix.is_finite() {
        Ok(())
    } else {
        Err("text matrix became non-finite while advancing glyphs".to_owned())
    }
}

fn load_embedded_font(
    doc: &Document,
    page_id: ObjectId,
    font_name: &[u8],
) -> Result<EmbeddedFont, String> {
    let page_fonts = doc
        .get_page_fonts(page_id)
        .map_err(|error| format!("read page font resources: {error}"))?;
    let font = page_fonts.get(font_name).ok_or_else(|| {
        format!(
            "page has no font resource /{}",
            String::from_utf8_lossy(font_name)
        )
    })?;
    let subtype = font
        .get(b"Subtype")
        .and_then(Object::as_name)
        .map_err(|error| format!("font has no valid /Subtype: {error}"))?;
    match subtype {
        b"Type0" => load_identity_h_font(doc, font),
        b"Type1" => load_simple_type1c_font(doc, font),
        other => Err(format!(
            "font subtype /{} is outside the admitted vector/text subset",
            String::from_utf8_lossy(other)
        )),
    }
}

fn load_identity_h_font(doc: &Document, type0: &Dictionary) -> Result<EmbeddedFont, String> {
    require_name(type0, b"Subtype", b"Type0", "font")?;
    require_name(type0, b"Encoding", b"Identity-H", "Type0 font")?;
    let descendants = type0
        .get(b"DescendantFonts")
        .and_then(Object::as_array)
        .map_err(|error| format!("Type0 /DescendantFonts is invalid: {error}"))?;
    let [descendant] = descendants.as_slice() else {
        return Err(format!(
            "Type0 font must have exactly one descendant, found {}",
            descendants.len()
        ));
    };
    let descendant = dereference_dict(doc, descendant, "CIDFont descendant")?;
    require_name(descendant, b"Subtype", b"CIDFontType2", "descendant font")?;
    require_name(descendant, b"CIDToGIDMap", b"Identity", "CIDFontType2")?;
    let widths = parse_cid_widths(doc, descendant)?;
    let descriptor_object = descendant
        .get(b"FontDescriptor")
        .map_err(|error| format!("CIDFontType2 has no /FontDescriptor: {error}"))?;
    let descriptor = dereference_dict(doc, descriptor_object, "FontDescriptor")?;
    let font_file_object = descriptor
        .get(b"FontFile2")
        .map_err(|error| format!("FontDescriptor has no /FontFile2: {error}"))?;
    let (_, font_file_object) = doc
        .dereference(font_file_object)
        .map_err(|error| format!("dereference FontFile2: {error}"))?;
    let stream = font_file_object
        .as_stream()
        .map_err(|error| format!("FontFile2 is not a stream: {error}"))?;
    if stream.dict.has(b"Filter") {
        return Err("compressed FontFile2 is outside the admitted MTDT subset".to_owned());
    }
    if stream.content.len() > MAX_VECTOR_FONT_BYTES {
        return Err(format!(
            "FontFile2 has {} bytes, exceeding the {MAX_VECTOR_FONT_BYTES}-byte limit",
            stream.content.len()
        ));
    }
    Face::parse(&stream.content, 0)
        .map_err(|error| format!("FontFile2 is not a valid bounded TrueType font: {error:?}"))?;
    Ok(EmbeddedFont::IdentityHTrueType {
        bytes: stream.content.clone(),
        widths,
    })
}

pub(super) fn parse_cid_widths(
    doc: &Document,
    descendant: &Dictionary,
) -> Result<CidWidths, String> {
    let default = match descendant.get(b"DW") {
        Ok(value) => number(value)?,
        Err(_) => 1000.0,
    };
    if !default.is_finite() || default < 0.0 {
        return Err(format!(
            "CIDFontType2 /DW {default} is negative or non-finite"
        ));
    }
    let Some(widths_object) = descendant.get(b"W").ok() else {
        return Ok(CidWidths {
            default,
            explicit: BTreeMap::new(),
        });
    };
    let (_, widths_object) = doc
        .dereference(widths_object)
        .map_err(|error| format!("dereference CIDFontType2 /W: {error}"))?;
    let widths = widths_object
        .as_array()
        .map_err(|error| format!("CIDFontType2 /W is not an array: {error}"))?;
    let mut explicit = BTreeMap::new();
    let mut index = 0usize;
    while index < widths.len() {
        let first = cid_integer(&widths[index], "CIDFontType2 /W start CID")?;
        let next = widths
            .get(index + 1)
            .ok_or_else(|| "CIDFontType2 /W ends after a start CID".to_owned())?;
        if let Object::Array(run) = next {
            if run.is_empty() {
                return Err("CIDFontType2 /W contains an empty width run".to_owned());
            }
            for (offset, width) in run.iter().enumerate() {
                let cid = usize::from(first)
                    .checked_add(offset)
                    .and_then(|value| u16::try_from(value).ok())
                    .ok_or_else(|| "CIDFontType2 /W width run exceeds CID 65535".to_owned())?;
                insert_cid_width(&mut explicit, cid, width)?;
            }
            index += 2;
        } else {
            let last = cid_integer(next, "CIDFontType2 /W end CID")?;
            if last < first {
                return Err(format!(
                    "CIDFontType2 /W range {first}..={last} is reversed"
                ));
            }
            let width = widths
                .get(index + 2)
                .ok_or_else(|| "CIDFontType2 /W range has no width".to_owned())?;
            for cid in first..=last {
                insert_cid_width(&mut explicit, cid, width)?;
            }
            index += 3;
        }
    }
    Ok(CidWidths { default, explicit })
}

fn cid_integer(object: &Object, role: &str) -> Result<u16, String> {
    let Object::Integer(value) = object else {
        return Err(format!("{role} is not an integer"));
    };
    u16::try_from(*value).map_err(|_| format!("{role} {value} is outside 0..=65535"))
}

fn insert_cid_width(
    widths: &mut BTreeMap<u16, f64>,
    cid: u16,
    object: &Object,
) -> Result<(), String> {
    let width = number(object)?;
    if !width.is_finite() || width < 0.0 {
        return Err(format!(
            "CIDFontType2 /W width for CID {cid} is negative or non-finite"
        ));
    }
    if widths.insert(cid, width).is_some() {
        return Err(format!("CIDFontType2 /W assigns CID {cid} more than once"));
    }
    Ok(())
}

fn load_simple_type1c_font(doc: &Document, type1: &Dictionary) -> Result<EmbeddedFont, String> {
    let program = load_simple_type1c_program(doc, type1)?;
    let widths = parse_simple_type1_widths(type1)?;
    Ok(EmbeddedFont::SimpleType1C {
        bytes: program.bytes,
        glyph_names: program.glyph_names,
        widths,
    })
}

pub(super) fn load_simple_type1c_program(
    doc: &Document,
    type1: &Dictionary,
) -> Result<SimpleType1CProgram, String> {
    require_name(type1, b"Subtype", b"Type1", "font")?;
    let glyph_names = parse_simple_type1_encoding(doc, type1)?;
    let descriptor_object = type1
        .get(b"FontDescriptor")
        .map_err(|error| format!("Type1 font has no /FontDescriptor: {error}"))?;
    let descriptor = dereference_dict(doc, descriptor_object, "Type1 FontDescriptor")?;
    let font_file_object = descriptor
        .get(b"FontFile3")
        .map_err(|error| format!("Type1 FontDescriptor has no /FontFile3: {error}"))?;
    let (_, font_file_object) = doc
        .dereference(font_file_object)
        .map_err(|error| format!("dereference Type1 FontFile3: {error}"))?;
    let stream = font_file_object
        .as_stream()
        .map_err(|error| format!("Type1 FontFile3 is not a stream: {error}"))?;
    require_name(&stream.dict, b"Subtype", b"Type1C", "FontFile3")?;
    if stream.content.len() > MAX_VECTOR_FONT_BYTES {
        return Err(format!(
            "encoded FontFile3 has {} bytes, exceeding the {MAX_VECTOR_FONT_BYTES}-byte limit",
            stream.content.len()
        ));
    }
    let bytes = match stream.dict.get(b"Filter") {
        Err(_) => stream.content.clone(),
        Ok(Object::Name(filter)) if filter == b"FlateDecode" => {
            if stream.dict.has(b"DecodeParms") || stream.dict.has(b"DP") {
                return Err("Type1C FontFile3 FlateDecode parameters are not admitted".to_owned());
            }
            bounded_font_inflate(&stream.content)?
        }
        Ok(Object::Name(filter)) => {
            return Err(format!(
                "Type1C FontFile3 filter /{} is not admitted",
                String::from_utf8_lossy(filter)
            ));
        }
        Ok(_) => return Err("Type1C FontFile3 /Filter is not a name".to_owned()),
    };
    cff::Table::parse(&bytes)
        .ok_or_else(|| "FontFile3 is not a valid bounded Type1C program".to_owned())?;
    Ok(SimpleType1CProgram { bytes, glyph_names })
}

pub(super) fn parse_simple_type1_encoding(
    doc: &Document,
    type1: &Dictionary,
) -> Result<Vec<Option<String>>, String> {
    let encoding_object = type1
        .get(b"Encoding")
        .map_err(|error| format!("Type1 font has no /Encoding: {error}"))?;
    let (_, encoding_object) = doc
        .dereference(encoding_object)
        .map_err(|error| format!("dereference Type1 Encoding: {error}"))?;
    let differences: &[Object] = match encoding_object {
        Object::Name(name) if name == b"WinAnsiEncoding" => &[],
        Object::Name(name) => {
            return Err(format!(
                "Type1 Encoding name /{} is not admitted; expected /WinAnsiEncoding",
                String::from_utf8_lossy(name)
            ));
        }
        Object::Dictionary(encoding) => {
            require_name(encoding, b"Type", b"Encoding", "Type1 Encoding")?;
            require_name(
                encoding,
                b"BaseEncoding",
                b"WinAnsiEncoding",
                "Type1 Encoding",
            )?;
            match encoding.get(b"Differences") {
                Ok(object) => object
                    .as_array()
                    .map_err(|error| format!("Type1 Encoding /Differences is invalid: {error}"))?
                    .as_slice(),
                Err(_) => &[],
            }
        }
        _ => {
            return Err(
                "Type1 Encoding must be /WinAnsiEncoding or an encoding dictionary".to_owned(),
            );
        }
    };
    let mut glyph_names = vec![None; 256];
    for code in 32u8..=255 {
        if let Some(name) = win_ansi_glyph_name(code) {
            glyph_names[usize::from(code)] = Some(name.to_owned());
        }
    }

    let mut next_code = None::<u16>;
    let mut explicitly_assigned = [false; 256];
    for item in differences {
        match item {
            Object::Integer(code) => {
                let code = u16::try_from(*code).map_err(|_| {
                    format!("Type1 Encoding /Differences code {code} is outside 0..=255")
                })?;
                if code > 255 {
                    return Err(format!(
                        "Type1 Encoding /Differences code {code} is outside 0..=255"
                    ));
                }
                next_code = Some(code);
            }
            Object::Name(name) => {
                let code = next_code.ok_or_else(|| {
                    "Type1 Encoding /Differences has a name before its first code".to_owned()
                })?;
                if code > 255 {
                    return Err(
                        "Type1 Encoding /Differences name sequence exceeds code 255".to_owned()
                    );
                }
                let index = usize::from(code);
                if explicitly_assigned[index] {
                    return Err(format!(
                        "Type1 Encoding /Differences assigns code {code} more than once"
                    ));
                }
                let name = std::str::from_utf8(name).map_err(|_| {
                    format!("Type1 Encoding /Differences code {code} has a non-UTF-8 glyph name")
                })?;
                if name.is_empty() || name.len() > MAX_TYPE1_GLYPH_NAME_BYTES {
                    return Err(format!(
                        "Type1 Encoding /Differences code {code} glyph name has {} bytes; admitted range is 1..={MAX_TYPE1_GLYPH_NAME_BYTES}",
                        name.len()
                    ));
                }
                glyph_names[index] = Some(name.to_owned());
                explicitly_assigned[index] = true;
                next_code = Some(code + 1);
            }
            _ => {
                return Err(
                    "Type1 Encoding /Differences contains neither an integer nor a name".to_owned(),
                );
            }
        }
    }
    Ok(glyph_names)
}

pub(super) fn win_ansi_glyph_name(code: u8) -> Option<&'static str> {
    const CONTROL: [&str; 33] = [
        "bullet",
        "Euro",
        "bullet",
        "quotesinglbase",
        "florin",
        "quotedblbase",
        "ellipsis",
        "dagger",
        "daggerdbl",
        "circumflex",
        "perthousand",
        "Scaron",
        "guilsinglleft",
        "OE",
        "bullet",
        "Zcaron",
        "bullet",
        "bullet",
        "quoteleft",
        "quoteright",
        "quotedblleft",
        "quotedblright",
        "bullet",
        "endash",
        "emdash",
        "tilde",
        "trademark",
        "scaron",
        "guilsinglright",
        "oe",
        "bullet",
        "zcaron",
        "Ydieresis",
    ];
    const LATIN: [&str; 96] = [
        "space",
        "exclamdown",
        "cent",
        "sterling",
        "currency",
        "yen",
        "brokenbar",
        "section",
        "dieresis",
        "copyright",
        "ordfeminine",
        "guillemotleft",
        "logicalnot",
        "hyphen",
        "registered",
        "macron",
        "degree",
        "plusminus",
        "twosuperior",
        "threesuperior",
        "acute",
        "mu",
        "paragraph",
        "periodcentered",
        "cedilla",
        "onesuperior",
        "ordmasculine",
        "guillemotright",
        "onequarter",
        "onehalf",
        "threequarters",
        "questiondown",
        "Agrave",
        "Aacute",
        "Acircumflex",
        "Atilde",
        "Adieresis",
        "Aring",
        "AE",
        "Ccedilla",
        "Egrave",
        "Eacute",
        "Ecircumflex",
        "Edieresis",
        "Igrave",
        "Iacute",
        "Icircumflex",
        "Idieresis",
        "Eth",
        "Ntilde",
        "Ograve",
        "Oacute",
        "Ocircumflex",
        "Otilde",
        "Odieresis",
        "multiply",
        "Oslash",
        "Ugrave",
        "Uacute",
        "Ucircumflex",
        "Udieresis",
        "Yacute",
        "Thorn",
        "germandbls",
        "agrave",
        "aacute",
        "acircumflex",
        "atilde",
        "adieresis",
        "aring",
        "ae",
        "ccedilla",
        "egrave",
        "eacute",
        "ecircumflex",
        "edieresis",
        "igrave",
        "iacute",
        "icircumflex",
        "idieresis",
        "eth",
        "ntilde",
        "ograve",
        "oacute",
        "ocircumflex",
        "otilde",
        "odieresis",
        "divide",
        "oslash",
        "ugrave",
        "uacute",
        "ucircumflex",
        "udieresis",
        "yacute",
        "thorn",
        "ydieresis",
    ];
    match code {
        32..=126 => Some(win_ansi_ascii_glyph_name(code)),
        127..=159 => Some(CONTROL[usize::from(code - 127)]),
        160..=255 => Some(LATIN[usize::from(code - 160)]),
        _ => None,
    }
}

fn win_ansi_ascii_glyph_name(code: u8) -> &'static str {
    match code {
        b' ' => "space",
        b'!' => "exclam",
        b'\"' => "quotedbl",
        b'#' => "numbersign",
        b'$' => "dollar",
        b'%' => "percent",
        b'&' => "ampersand",
        b'\'' => "quotesingle",
        b'(' => "parenleft",
        b')' => "parenright",
        b'*' => "asterisk",
        b'+' => "plus",
        b',' => "comma",
        b'-' => "hyphen",
        b'.' => "period",
        b'/' => "slash",
        b'0' => "zero",
        b'1' => "one",
        b'2' => "two",
        b'3' => "three",
        b'4' => "four",
        b'5' => "five",
        b'6' => "six",
        b'7' => "seven",
        b'8' => "eight",
        b'9' => "nine",
        b':' => "colon",
        b';' => "semicolon",
        b'<' => "less",
        b'=' => "equal",
        b'>' => "greater",
        b'?' => "question",
        b'@' => "at",
        b'A' => "A",
        b'B' => "B",
        b'C' => "C",
        b'D' => "D",
        b'E' => "E",
        b'F' => "F",
        b'G' => "G",
        b'H' => "H",
        b'I' => "I",
        b'J' => "J",
        b'K' => "K",
        b'L' => "L",
        b'M' => "M",
        b'N' => "N",
        b'O' => "O",
        b'P' => "P",
        b'Q' => "Q",
        b'R' => "R",
        b'S' => "S",
        b'T' => "T",
        b'U' => "U",
        b'V' => "V",
        b'W' => "W",
        b'X' => "X",
        b'Y' => "Y",
        b'Z' => "Z",
        b'[' => "bracketleft",
        b'\\' => "backslash",
        b']' => "bracketright",
        b'^' => "asciicircum",
        b'_' => "underscore",
        b'`' => "grave",
        b'a' => "a",
        b'b' => "b",
        b'c' => "c",
        b'd' => "d",
        b'e' => "e",
        b'f' => "f",
        b'g' => "g",
        b'h' => "h",
        b'i' => "i",
        b'j' => "j",
        b'k' => "k",
        b'l' => "l",
        b'm' => "m",
        b'n' => "n",
        b'o' => "o",
        b'p' => "p",
        b'q' => "q",
        b'r' => "r",
        b's' => "s",
        b't' => "t",
        b'u' => "u",
        b'v' => "v",
        b'w' => "w",
        b'x' => "x",
        b'y' => "y",
        b'z' => "z",
        b'{' => "braceleft",
        b'|' => "bar",
        b'}' => "braceright",
        b'~' => "asciitilde",
        _ => unreachable!("caller only supplies printable ASCII"),
    }
}

pub(super) fn parse_simple_type1_widths(type1: &Dictionary) -> Result<Vec<Option<f64>>, String> {
    let first = type1
        .get(b"FirstChar")
        .and_then(Object::as_i64)
        .map_err(|error| format!("Type1 /FirstChar is invalid: {error}"))?;
    let last = type1
        .get(b"LastChar")
        .and_then(Object::as_i64)
        .map_err(|error| format!("Type1 /LastChar is invalid: {error}"))?;
    if !(0..=255).contains(&first) || !(first..=255).contains(&last) {
        return Err(format!(
            "Type1 character range {first}..={last} is outside 0..=255 or reversed"
        ));
    }
    let declared = type1
        .get(b"Widths")
        .and_then(Object::as_array)
        .map_err(|error| format!("Type1 /Widths is invalid: {error}"))?;
    let expected =
        usize::try_from(last - first + 1).map_err(|_| "Type1 width-count overflow".to_owned())?;
    if declared.len() != expected {
        return Err(format!(
            "Type1 /Widths has {} entries, expected {expected} for {first}..={last}",
            declared.len()
        ));
    }
    let mut widths = vec![None; 256];
    for (offset, object) in declared.iter().enumerate() {
        let width = number(object)?;
        if !width.is_finite() || width < 0.0 {
            return Err(format!(
                "Type1 /Widths entry {offset} is negative or non-finite"
            ));
        }
        let code = usize::try_from(first)
            .ok()
            .and_then(|first| first.checked_add(offset))
            .ok_or_else(|| "Type1 width code overflow".to_owned())?;
        widths[code] = Some(width);
    }
    Ok(widths)
}

fn bounded_font_inflate(raw: &[u8]) -> Result<Vec<u8>, String> {
    use std::io::Read;

    let mut bytes = Vec::new();
    flate2::read::ZlibDecoder::new(raw)
        .take((MAX_VECTOR_FONT_BYTES as u64).saturating_add(1))
        .read_to_end(&mut bytes)
        .map_err(|error| format!("inflate Type1C FontFile3: {error}"))?;
    if bytes.len() > MAX_VECTOR_FONT_BYTES {
        return Err(format!(
            "inflated FontFile3 exceeds the {MAX_VECTOR_FONT_BYTES}-byte limit"
        ));
    }
    Ok(bytes)
}

fn dereference_dict<'a>(
    doc: &'a Document,
    object: &'a Object,
    role: &str,
) -> Result<&'a Dictionary, String> {
    let (_, object) = doc
        .dereference(object)
        .map_err(|error| format!("dereference {role}: {error}"))?;
    object
        .as_dict()
        .map_err(|error| format!("{role} is not a dictionary: {error}"))
}

fn require_name(dict: &Dictionary, key: &[u8], expected: &[u8], role: &str) -> Result<(), String> {
    let actual = dict.get(key).and_then(Object::as_name).map_err(|error| {
        format!(
            "{role} has no valid /{} name: {error}",
            String::from_utf8_lossy(key)
        )
    })?;
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "{role} /{} is /{}, expected /{}",
            String::from_utf8_lossy(key),
            String::from_utf8_lossy(actual),
            String::from_utf8_lossy(expected)
        ))
    }
}

fn set_text_font(text: &mut TextState, operands: &[Object]) -> Result<(), String> {
    let [Object::Name(name), size] = operands else {
        return Err("Tf expects a font name and size".to_owned());
    };
    let size = number(size)?;
    if !size.is_finite() || size <= 0.0 {
        return Err(format!("invalid Tf size {size}"));
    }
    text.font_name = Some(name.clone());
    text.font_size = size;
    Ok(())
}

fn translate_text_matrix(text: &mut TextState, operands: &[Object]) -> Result<(), String> {
    let point = point_operands(operands, "Td")?;
    text.line_matrix = text.line_matrix.concat(Matrix {
        a: 1.0,
        b: 0.0,
        c: 0.0,
        d: 1.0,
        e: point.x,
        f: point.y,
    });
    text.matrix = text.line_matrix;
    if text.line_matrix.is_finite() {
        Ok(())
    } else {
        Err("Td produced a non-finite text matrix".to_owned())
    }
}

fn set_text_leading(text: &mut TextState, operands: &[Object]) -> Result<(), String> {
    text.leading = one_number(operands, "TL")?;
    Ok(())
}

fn next_text_line(text: &mut TextState) -> Result<(), String> {
    text.line_matrix = text.line_matrix.concat(Matrix {
        a: 1.0,
        b: 0.0,
        c: 0.0,
        d: 1.0,
        e: 0.0,
        f: -text.leading,
    });
    text.matrix = text.line_matrix;
    if text.line_matrix.is_finite() {
        Ok(())
    } else {
        Err("text next-line operation produced a non-finite matrix".to_owned())
    }
}

fn line_cap_operand(operands: &[Object]) -> Result<LineCap, String> {
    match one_number(operands, "J")? {
        0.0 => Ok(LineCap::Butt),
        1.0 => Ok(LineCap::Round),
        2.0 => Ok(LineCap::Square),
        value => Err(format!("J line-cap value {value} is outside 0..=2")),
    }
}

fn line_join_operand(operands: &[Object]) -> Result<LineJoin, String> {
    match one_number(operands, "j")? {
        0.0 => Ok(LineJoin::Miter),
        1.0 => Ok(LineJoin::Round),
        2.0 => Ok(LineJoin::Bevel),
        value => Err(format!("j line-join value {value} is outside 0..=2")),
    }
}

fn append_rectangle(path: &mut PaintPath, ctm: Matrix, operands: &[Object]) -> Result<(), String> {
    let [x, y, width, height] = operands else {
        return Err("re expects four numeric operands".to_owned());
    };
    let x = number(x)?;
    let y = number(y)?;
    let width = number(width)?;
    let height = number(height)?;
    let right = x + width;
    let top = y + height;
    if ![x, y, width, height, right, top]
        .into_iter()
        .all(f64::is_finite)
    {
        return Err("re contains a non-finite coordinate or extent".to_owned());
    }
    path.move_to(ctm.point(Point { x, y }))?;
    path.line_to(ctm.point(Point { x: right, y }))?;
    path.line_to(ctm.point(Point { x: right, y: top }))?;
    path.line_to(ctm.point(Point { x, y: top }))?;
    path.close()
}

pub(super) fn apply_lilypond_ext_gstate(
    doc: &Document,
    page_id: ObjectId,
    operands: &[Object],
) -> Result<(), String> {
    let [Object::Name(name)] = operands else {
        return Err("gs expects one graphics-state resource name".to_owned());
    };
    let (direct_resources, resource_ids) = doc
        .get_page_resources(page_id)
        .map_err(|error| format!("read page resources for gs: {error}"))?;

    let mut state = None;
    if let Some(resources) = direct_resources {
        state = find_ext_gstate(doc, resources, name)?;
    }
    if state.is_none() {
        for resource_id in resource_ids {
            let resources = doc
                .get_dictionary(resource_id)
                .map_err(|error| format!("read inherited page resources for gs: {error}"))?;
            if let Some(found) = find_ext_gstate(doc, resources, name)? {
                state = Some(found);
                break;
            }
        }
    }
    let state = state.ok_or_else(|| {
        format!(
            "page has no ExtGState resource /{}",
            String::from_utf8_lossy(name)
        )
    })?;
    for (key, _) in state.iter() {
        if key != b"Type" && key != b"SA" {
            return Err(format!(
                "ExtGState /{} contains unsupported key /{}",
                String::from_utf8_lossy(name),
                String::from_utf8_lossy(key)
            ));
        }
    }
    require_name(state, b"Type", b"ExtGState", "LilyPond ExtGState")?;
    let stroke_adjustment = state
        .get(b"SA")
        .and_then(Object::as_bool)
        .map_err(|error| format!("LilyPond ExtGState /SA is invalid: {error}"))?;
    if stroke_adjustment {
        return Err("ExtGState stroke adjustment /SA true is not admitted".to_owned());
    }
    Ok(())
}

fn find_ext_gstate<'a>(
    doc: &'a Document,
    resources: &'a Dictionary,
    name: &[u8],
) -> Result<Option<&'a Dictionary>, String> {
    let ext_gstates = match resources.get(b"ExtGState") {
        Ok(object) => dereference_dict(doc, object, "page ExtGState resources")?,
        Err(_) => return Ok(None),
    };
    let state = match ext_gstates.get(name) {
        Ok(object) => dereference_dict(doc, object, "named ExtGState")?,
        Err(_) => return Ok(None),
    };
    Ok(Some(state))
}

fn require_black_gray(operands: &[Object]) -> Result<(), String> {
    let value = one_number(operands, "gray color")?;
    if value.abs() <= f64::EPSILON {
        Ok(())
    } else {
        Err(format!("non-black gray value {value} is not admitted yet"))
    }
}

fn require_black_rgb(operands: &[Object]) -> Result<(), String> {
    if operands.len() != 3 {
        return Err("RGB color expects three operands".to_owned());
    }
    let values = operands.iter().map(number).collect::<Result<Vec<_>, _>>()?;
    if values.iter().all(|value| value.abs() <= f64::EPSILON) {
        Ok(())
    } else {
        Err(format!(
            "non-black RGB value {values:?} is not admitted yet"
        ))
    }
}

fn number(object: &Object) -> Result<f64, String> {
    match object {
        Object::Integer(value) => Ok(*value as f64),
        Object::Real(value) => Ok(f64::from(*value)),
        _ => Err("PDF operand is not numeric".to_owned()),
    }
}

fn one_number(operands: &[Object], operator: &str) -> Result<f64, String> {
    let [value] = operands else {
        return Err(format!("{operator} expects one numeric operand"));
    };
    let value = number(value)?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!("{operator} operand is not finite"))
    }
}

fn require_no_operands(operands: &[Object], operator: &str) -> Result<(), String> {
    if operands.is_empty() {
        Ok(())
    } else {
        Err(format!("{operator} expects no operands"))
    }
}

fn point_operands(operands: &[Object], operator: &str) -> Result<Point, String> {
    let [x, y] = operands else {
        return Err(format!("{operator} expects two numeric operands"));
    };
    let point = Point {
        x: number(x)?,
        y: number(y)?,
    };
    if point.x.is_finite() && point.y.is_finite() {
        Ok(point)
    } else {
        Err(format!("{operator} point is not finite"))
    }
}

fn cubic_operands(operands: &[Object]) -> Result<[Point; 3], String> {
    let [x1, y1, x2, y2, x3, y3] = operands else {
        return Err("c expects six numeric operands".to_owned());
    };
    let points = [
        Point {
            x: number(x1)?,
            y: number(y1)?,
        },
        Point {
            x: number(x2)?,
            y: number(y2)?,
        },
        Point {
            x: number(x3)?,
            y: number(y3)?,
        },
    ];
    if points
        .iter()
        .all(|point| point.x.is_finite() && point.y.is_finite())
    {
        Ok(points)
    } else {
        Err("c contains a non-finite coordinate".to_owned())
    }
}

fn matrix_operands(operands: &[Object]) -> Result<Matrix, String> {
    let [a, b, c, d, e, f] = operands else {
        return Err("matrix operator expects six numeric operands".to_owned());
    };
    let matrix = Matrix {
        a: number(a)?,
        b: number(b)?,
        c: number(c)?,
        d: number(d)?,
        e: number(e)?,
        f: number(f)?,
    };
    if matrix.is_finite() {
        Ok(matrix)
    } else {
        Err("matrix contains a non-finite value".to_owned())
    }
}

fn dash_operands(operands: &[Object]) -> Result<(Vec<f64>, f64), String> {
    let [Object::Array(pattern), phase] = operands else {
        return Err("d expects an array and phase".to_owned());
    };
    if pattern.len() > 64 {
        return Err("dash array exceeds 64 entries".to_owned());
    }
    let mut pattern = pattern.iter().map(number).collect::<Result<Vec<_>, _>>()?;
    if pattern
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err("dash array has a negative or non-finite entry".to_owned());
    }
    if !pattern.is_empty() && pattern.iter().all(|value| *value == 0.0) {
        return Err("dash array cannot contain only zero lengths".to_owned());
    }
    if pattern.len() % 2 == 1 {
        pattern.extend_from_within(..);
    }
    let phase = number(phase)?;
    if !phase.is_finite() {
        return Err("dash phase is non-finite".to_owned());
    }
    Ok((pattern, phase))
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use lopdf::{Stream, StringFormat, dictionary};

    use super::*;
    use crate::pdf::PdfPages;

    fn operation(operator: &str, operands: Vec<Object>) -> Operation {
        Operation::new(operator, operands)
    }

    fn minimal_type1c_program() -> Vec<u8> {
        // A complete two-glyph CFF: .notdef plus /A, whose Type2 charstring is
        // a 500x700 rectangle. The PDF Encoding maps binary code 0 to /A.
        vec![
            1, 0, 4, 4, // header
            0, 1, 1, 1, 2, b'T', // Name INDEX
            0, 1, 1, 1, 5, 162, 15, 165, 17, // Top DICT: charset=23, CharStrings=26
            0, 0, // String INDEX
            0, 0, // Global Subr INDEX
            0, 0, 34, // format-0 charset: GID 1 is standard SID 34 (/A)
            0, 2, 1, 1, 2, 19, // CharStrings INDEX header and offsets
            14, // .notdef: endchar
            139, 139, 21, // /A: 0 0 rmoveto
            248, 136, 139, 139, 249, 80, 252, 136, 139, 139, 253, 80, 5,  // rectangle rlineto
            14, // endchar
        ]
    }

    fn type1c_pdf(extra_ext_gstate_key: bool) -> Vec<u8> {
        let cff = minimal_type1c_program();
        assert!(
            cff::Table::parse(&cff).is_some(),
            "constructed CFF is invalid"
        );
        let mut encoder =
            flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(&cff).expect("compress Type1C fixture");
        let compressed = encoder.finish().expect("finish Type1C compression");

        let mut doc = Document::with_version("1.7");
        let pages_id = doc.new_object_id();
        let font_file_id = doc.add_object(Stream::new(
            dictionary! {
                "Filter" => "FlateDecode",
                "Subtype" => "Type1C",
            },
            compressed,
        ));
        let descriptor_id = doc.add_object(dictionary! {
            "Type" => "FontDescriptor",
            "FontName" => "FixtureType1C",
            "FontFile3" => font_file_id,
        });
        let encoding_id = doc.add_object(dictionary! {
            "Type" => "Encoding",
            "BaseEncoding" => "WinAnsiEncoding",
            "Differences" => vec![0.into(), Object::Name(b"A".to_vec())],
        });
        let font_id = doc.add_object(dictionary! {
            "Type" => "Font",
            "Subtype" => "Type1",
            "BaseFont" => "FixtureType1C",
            "FirstChar" => 0,
            "LastChar" => 0,
            "Widths" => vec![500.into()],
            "Encoding" => encoding_id,
            "FontDescriptor" => descriptor_id,
        });
        let mut ext_gstate = dictionary! {
            "Type" => "ExtGState",
            "SA" => false,
        };
        if extra_ext_gstate_key {
            ext_gstate.set("ca", 1);
        }
        let ext_gstate_id = doc.add_object(ext_gstate);
        let resources_id = doc.add_object(dictionary! {
            "Font" => dictionary! { "F1" => font_id },
            "ExtGState" => dictionary! { "GS1" => ext_gstate_id },
        });
        let glyph = || Object::String(vec![0], StringFormat::Hexadecimal);
        let content = Content {
            operations: vec![
                operation("q", vec![]),
                operation("gs", vec![Object::Name(b"GS1".to_vec())]),
                operation("J", vec![1.into()]),
                operation("j", vec![1.into()]),
                operation("w", vec![2.into()]),
                operation("re", vec![10.into(), 10.into(), 30.into(), 10.into()]),
                operation("S", vec![]),
                operation("Q", vec![]),
                operation("BT", vec![]),
                operation("Tf", vec![Object::Name(b"F1".to_vec()), 20.into()]),
                operation("TL", vec![20.into()]),
                operation(
                    "Tm",
                    vec![1.into(), 0.into(), 0.into(), 1.into(), 20.into(), 60.into()],
                ),
                operation("TJ", vec![Object::Array(vec![glyph()])]),
                operation("ET", vec![]),
                operation("q", vec![]),
                operation("BT", vec![]),
                operation("TL", vec![10.into()]),
                operation("ET", vec![]),
                operation("Q", vec![]),
                operation("BT", vec![]),
                operation(
                    "Tm",
                    vec![1.into(), 0.into(), 0.into(), 1.into(), 20.into(), 60.into()],
                ),
                operation("'", vec![glyph()]),
                operation("ET", vec![]),
            ],
        }
        .encode()
        .expect("encode vector fixture");
        let content_id = doc.add_object(Stream::new(Dictionary::new(), content));
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "Resources" => resources_id,
            "Contents" => content_id,
            "MediaBox" => vec![0.into(), 0.into(), 100.into(), 100.into()],
        });
        doc.objects.insert(
            pages_id,
            Object::Dictionary(dictionary! {
                "Type" => "Pages",
                "Kids" => vec![page_id.into()],
                "Count" => 1,
            }),
        );
        let catalog_id = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
        doc.trailer.set("Root", catalog_id);
        let mut bytes = Vec::new();
        doc.save_to(&mut bytes).expect("save vector fixture");
        bytes
    }

    #[test]
    fn filled_path_and_stroke_render_nonblank_at_pdf_coordinates() {
        let content = Content {
            operations: vec![
                operation("m", vec![10.into(), 10.into()]),
                operation("l", vec![30.into(), 10.into()]),
                operation("l", vec![30.into(), 30.into()]),
                operation("l", vec![10.into(), 30.into()]),
                operation("h", vec![]),
                operation("f", vec![]),
                operation("w", vec![2.into()]),
                operation("m", vec![5.into(), 40.into()]),
                operation("l", vec![45.into(), 40.into()]),
                operation("S", vec![]),
            ],
        };
        let image =
            render_mtdt_vector_page(&Document::new(), (1, 0), &content, [0.0, 0.0, 50.0, 50.0])
                .expect("render admitted paths");
        let gray = image.to_luma8();
        assert_eq!(gray.dimensions(), (50, 50));
        assert!(gray.get_pixel(20, 30).0[0] < 32, "filled square missing");
        assert!(gray.get_pixel(20, 10).0[0] < 200, "stroked line missing");
        assert_eq!(gray.get_pixel(2, 2).0[0], 255, "background is not white");
    }

    #[test]
    fn unknown_operator_refuses_with_exact_operation_context() {
        let content = Content {
            operations: vec![operation("sh", vec![])],
        };
        let error =
            render_mtdt_vector_page(&Document::new(), (1, 0), &content, [0.0, 0.0, 10.0, 10.0])
                .expect_err("unknown paint must refuse");
        assert!(error.contains("operation 0 (sh)"), "{error}");
        assert!(error.contains("outside the admitted MTDT"), "{error}");
    }

    #[test]
    fn page_pixel_budget_fails_before_allocation() {
        let error = SupersampledRaster::new([0.0, 0.0, 10_000.0, 10_000.0])
            .err()
            .expect("oversized page must refuse");
        assert!(error.contains("output pixels"), "{error}");
    }

    #[test]
    fn cid_width_runs_use_pdf_defaults_ranges_and_arrays_without_overlap() {
        let descendant = dictionary! {
            "DW" => 700,
            "W" => vec![
                1.into(),
                Object::Array(vec![250.into(), 300.into()]),
                10.into(),
                12.into(),
                400.into(),
            ],
        };
        let widths = parse_cid_widths(&Document::new(), &descendant).expect("parse CID widths");
        assert_eq!(widths.get(0), 700.0);
        assert_eq!(widths.get(1), 250.0);
        assert_eq!(widths.get(2), 300.0);
        assert_eq!(widths.get(10), 400.0);
        assert_eq!(widths.get(12), 400.0);
        assert_eq!(widths.get(13), 700.0);

        let duplicate = dictionary! {
            "W" => vec![
                1.into(),
                Object::Array(vec![250.into(), 300.into()]),
                2.into(),
                2.into(),
                400.into(),
            ],
        };
        let error = parse_cid_widths(&Document::new(), &duplicate)
            .expect_err("duplicate CID widths must refuse");
        assert!(error.contains("assigns CID 2 more than once"), "{error}");
    }

    #[test]
    fn win_ansi_base_covers_extended_codes_before_differences_override() {
        let encoding = dictionary! {
            "Type" => "Encoding",
            "BaseEncoding" => "WinAnsiEncoding",
        };
        let font = dictionary! { "Encoding" => encoding };
        let names = parse_simple_type1_encoding(&Document::new(), &font)
            .expect("parse complete WinAnsi base");
        assert_eq!(names[0x7f].as_deref(), Some("bullet"));
        assert_eq!(names[0x80].as_deref(), Some("Euro"));
        assert_eq!(names[0x91].as_deref(), Some("quoteleft"));
        assert_eq!(names[0xa0].as_deref(), Some("space"));
        assert_eq!(names[0xff].as_deref(), Some("ydieresis"));
    }

    #[test]
    fn type1_named_win_ansi_encoding_accepts_direct_and_indirect_names_only() {
        let direct_font = dictionary! {
            "Encoding" => Object::Name(b"WinAnsiEncoding".to_vec()),
        };
        let direct = parse_simple_type1_encoding(&Document::new(), &direct_font)
            .expect("direct /WinAnsiEncoding name is a legal Type1 encoding");

        let mut indirect_doc = Document::with_version("1.7");
        let indirect_encoding = indirect_doc.add_object(Object::Name(b"WinAnsiEncoding".to_vec()));
        let indirect_font = dictionary! { "Encoding" => indirect_encoding };
        let indirect = parse_simple_type1_encoding(&indirect_doc, &indirect_font)
            .expect("indirect /WinAnsiEncoding name is a legal Type1 encoding");

        assert_eq!(direct, indirect);
        assert_eq!(direct[0x20].as_deref(), Some("space"));
        assert_eq!(direct[0x41].as_deref(), Some("A"));
        assert_eq!(direct[0x80].as_deref(), Some("Euro"));

        for wrong_name in [b"MacRomanEncoding".as_slice(), b"MacExpertEncoding"] {
            let wrong_font = dictionary! {
                "Encoding" => Object::Name(wrong_name.to_vec()),
            };
            let error = parse_simple_type1_encoding(&Document::new(), &wrong_font)
                .expect_err("unreviewed named Type1 encoding must refuse");
            assert!(error.contains("expected /WinAnsiEncoding"), "{error}");
        }

        let wrong_type = dictionary! { "Encoding" => Object::Integer(7) };
        let error = parse_simple_type1_encoding(&Document::new(), &wrong_type)
            .expect_err("non-name, non-dictionary Type1 encoding must refuse");
        assert!(error.contains("must be /WinAnsiEncoding"), "{error}");
    }

    #[test]
    fn stroke_semantics_handle_hairlines_scaled_dashes_and_refuse_shear() {
        let content = Content {
            operations: vec![
                operation("w", vec![0.into()]),
                operation("m", vec![1.into(), 5.into()]),
                operation("l", vec![9.into(), 5.into()]),
                operation("S", vec![]),
            ],
        };
        let image =
            render_mtdt_vector_page(&Document::new(), (1, 0), &content, [0.0, 0.0, 10.0, 10.0])
                .expect("render device hairline")
                .to_luma8();
        assert!(
            image.pixels().any(|pixel| pixel.0[0] < 128),
            "device hairline disappeared"
        );

        let graphics = GraphicsState {
            ctm: Matrix {
                a: 2.0,
                b: 0.0,
                c: 0.0,
                d: -2.0,
                e: 0.0,
                f: 0.0,
            },
            line_width: 2.0,
            dash: vec![1.0, 2.0],
            dash_phase: 1.0,
            ..GraphicsState::default()
        };
        let (width, dash, phase) =
            page_space_stroke_parameters(&graphics).expect("scale similarity stroke");
        assert_eq!(width, 4.0);
        assert_eq!(dash, vec![2.0, 4.0]);
        assert_eq!(phase, 2.0);

        let unsupported = Content {
            operations: vec![
                operation(
                    "cm",
                    vec![2.into(), 0.into(), 0.into(), 1.into(), 0.into(), 0.into()],
                ),
                operation("w", vec![1.into()]),
                operation("m", vec![1.into(), 5.into()]),
                operation("l", vec![9.into(), 5.into()]),
                operation("S", vec![]),
            ],
        };
        let error = render_mtdt_vector_page(
            &Document::new(),
            (1, 0),
            &unsupported,
            [0.0, 0.0, 20.0, 10.0],
        )
        .expect_err("anisotropic stroke must refuse");
        assert!(error.contains("operation 4 (S)"), "{error}");
        assert!(error.contains("anisotropic scale or shear"), "{error}");
    }

    #[test]
    fn odd_dash_arrays_double_and_text_operators_require_bt_et() {
        let (pattern, phase) = dash_operands(&[Object::Array(vec![3.into()]), 4.into()])
            .expect("normalize odd dash array");
        assert_eq!(pattern, vec![3.0, 3.0]);
        let cursor = DashCursor::new(&pattern, phase).expect("position dash cursor");
        assert!(!cursor.ink, "phase in doubled half must begin with a gap");
        assert_eq!(cursor.remaining, 2.0);
        let all_zero = dash_operands(&[Object::Array(vec![0.into()]), 0.into()])
            .expect_err("all-zero dash must refuse");
        assert!(all_zero.contains("only zero"), "{all_zero}");

        let content = Content {
            operations: vec![operation(
                "Tj",
                vec![Object::String(vec![0], StringFormat::Hexadecimal)],
            )],
        };
        let error =
            render_mtdt_vector_page(&Document::new(), (1, 0), &content, [0.0, 0.0, 10.0, 10.0])
                .expect_err("text show outside BT must refuse");
        assert!(error.contains("operation 0 (Tj)"), "{error}");
        assert!(error.contains("outside BT/ET"), "{error}");
    }

    #[test]
    fn text_marked_content_is_raster_neutral_but_strictly_balanced() {
        let content = Content {
            operations: vec![
                operation("BT", vec![]),
                operation(
                    "BDC",
                    vec![
                        Object::Name(b"Span".to_vec()),
                        dictionary! { "ActualText" => Object::string_literal("mf") }.into(),
                    ],
                ),
                operation("BMC", vec![Object::Name(b"Artifact".to_vec())]),
                operation("EMC", vec![]),
                operation("EMC", vec![]),
                operation("ET", vec![]),
            ],
        };
        render_mtdt_vector_page(&Document::new(), (1, 0), &content, [0.0, 0.0, 10.0, 10.0])
            .expect("balanced text marked content is metadata, not unsupported paint");

        let outside = Content {
            operations: vec![operation(
                "BDC",
                vec![Object::Name(b"Span".to_vec()), Dictionary::new().into()],
            )],
        };
        let error =
            render_mtdt_vector_page(&Document::new(), (1, 0), &outside, [0.0, 0.0, 10.0, 10.0])
                .expect_err("marked content outside BT/ET must remain outside the MTDT subset");
        assert!(error.contains("outside BT/ET"), "{error}");

        let unbalanced = Content {
            operations: vec![
                operation("BT", vec![]),
                operation("EMC", vec![]),
                operation("ET", vec![]),
            ],
        };
        let error = render_mtdt_vector_page(
            &Document::new(),
            (1, 0),
            &unbalanced,
            [0.0, 0.0, 10.0, 10.0],
        )
        .expect_err("unmatched EMC must refuse");
        assert!(error.contains("EMC has no matching"), "{error}");
    }

    #[test]
    fn cumulative_page_budget_counts_transient_paths_and_glyphs() {
        let path = PaintPath {
            contours: vec![Contour::default()],
            segment_count: 1,
        };
        let mut budget = RenderBudget {
            path_segments: MAX_VECTOR_PATH_SEGMENTS,
            ..RenderBudget::default()
        };
        let error = budget
            .charge_complete_path(&path)
            .expect_err("cumulative path overflow must refuse");
        assert!(error.contains("cumulative"), "{error}");

        let mut budget = RenderBudget {
            glyphs: MAX_VECTOR_GLYPHS,
            ..RenderBudget::default()
        };
        let error = budget
            .charge_glyph()
            .expect_err("cumulative glyph overflow must refuse");
        assert!(error.contains("glyph limit"), "{error}");
    }

    #[test]
    fn compressed_type1c_binary_codes_and_lilypond_operators_render_deterministically() {
        let pages = PdfPages::from_bytes(type1c_pdf(false)).expect("open constructed PDF");
        let first = pages
            .render(0)
            .expect("render constructed Type1C page")
            .to_luma8();
        let second = pages
            .render(0)
            .expect("replay constructed Type1C page")
            .to_luma8();
        assert_eq!(first.dimensions(), (100, 100));
        assert_eq!(first.as_raw(), second.as_raw(), "vector replay drifted");
        let first_glyph_ink = (20..30)
            .flat_map(|x| (25..41).map(move |y| (x, y)))
            .filter(|(x, y)| first.get_pixel(*x, *y).0[0] < 128)
            .count();
        let second_glyph_ink = (20..30)
            .flat_map(|x| (45..61).map(move |y| (x, y)))
            .filter(|(x, y)| first.get_pixel(*x, *y).0[0] < 128)
            .count();
        assert!(first_glyph_ink > 50, "TJ one-byte Type1C glyph is missing");
        assert!(
            second_glyph_ink > 50,
            "quote-operator Type1C glyph is missing"
        );
    }

    #[test]
    fn ext_gstate_with_unimplemented_paint_semantics_refuses_in_context() {
        let pages = PdfPages::from_bytes(type1c_pdf(true)).expect("open constructed PDF");
        let error = pages.render(0).expect_err("fill alpha must not be ignored");
        let message = error.to_string();
        assert!(message.contains("operation 1 (gs)"), "{message}");
        assert!(message.contains("unsupported key /ca"), "{message}");
    }
}
