import ActivityKit
import Foundation
import SwiftUI
import WidgetKit

private let visionGreen = Color(red: 0.20, green: 0.83, blue: 0.60)
private let visionGold = Color(red: 0.98, green: 0.72, blue: 0.20)
private let visionInk = Color(red: 0.004, green: 0.028, blue: 0.018)

struct OCRTimelineEntry: TimelineEntry {
    let date: Date
    let snapshot: FrankenOCRWidgetSnapshot
}

struct OCRTimelineProvider: TimelineProvider {
    func placeholder(in context: Context) -> OCRTimelineEntry {
        OCRTimelineEntry(date: .now, snapshot: .placeholder)
    }
    func getSnapshot(in context: Context, completion: @escaping (OCRTimelineEntry) -> Void) {
        completion(OCRTimelineEntry(date: .now, snapshot: FrankenOCRSharedStore.loadSnapshot()))
    }
    func getTimeline(in context: Context, completion: @escaping (Timeline<OCRTimelineEntry>) -> Void) {
        let entry = OCRTimelineEntry(date: .now, snapshot: FrankenOCRSharedStore.loadSnapshot())
        completion(Timeline(entries: [entry], policy: .after(.now.addingTimeInterval(15 * 60))))
    }
}

struct FrankenOCRVisionWidget: Widget {
    let kind = "FrankenOCRVisionWidget"
    var body: some WidgetConfiguration {
        StaticConfiguration(kind: kind, provider: OCRTimelineProvider()) { entry in
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Image(systemName: "viewfinder.circle")
                        .font(.title3.bold()).foregroundStyle(visionGold)
                    Spacer()
                    Text("READS_LOCALLY")
                        .font(.system(size: 8, weight: .black, design: .monospaced))
                        .tracking(1.3).foregroundStyle(visionGreen)
                }
                Spacer(minLength: 0)
                Text(entry.snapshot.headline).font(.headline).foregroundStyle(.white).lineLimit(2)
                Text(entry.snapshot.detail).font(.caption).foregroundStyle(.white.opacity(0.68)).lineLimit(2)
            }
            .containerBackground(for: .widget) {
                LinearGradient(
                    colors: [visionInk, Color.black, visionGold.opacity(0.12)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            }
            .widgetURL(URL(string: "frankenocr://recognize"))
        }
        .configurationDisplayName("Vision Table")
        .description("See private recognition status and open FrankenOCR.")
        .supportedFamilies([.systemSmall, .systemMedium])
    }
}

struct FrankenOCRLiveActivity: Widget {
    var body: some WidgetConfiguration {
        ActivityConfiguration(for: FrankenOCRRunActivityAttributes.self) { context in
            OCRLockView(context: context)
                .activityBackgroundTint(visionInk)
                .activitySystemActionForegroundColor(visionGold)
                .widgetURL(URL(string: "frankenocr://recognize"))
        } dynamicIsland: { context in
            DynamicIsland {
                DynamicIslandExpandedRegion(.leading) {
                    Image(systemName: icon(context.state.status)).font(.title2.bold()).foregroundStyle(visionGold)
                }
                DynamicIslandExpandedRegion(.trailing) {
                    OCRElapsedView(context: context)
                }
                DynamicIslandExpandedRegion(.center) {
                    Text(context.state.stage).font(.headline).lineLimit(1)
                }
                DynamicIslandExpandedRegion(.bottom) {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack(alignment: .firstTextBaseline) {
                            Text(context.state.detail).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                            Spacer(minLength: 8)
                            if context.state.status != .complete && context.state.status != .failed {
                                Link("Stop", destination: URL(string: "frankenocr://cancel")!)
                                    .font(.caption.weight(.semibold)).foregroundStyle(.red)
                            }
                        }
                        OCRUnitRail(state: context.state)
                    }
                }
            } compactLeading: {
                Image(systemName: icon(context.state.status)).foregroundStyle(visionGold)
            } compactTrailing: {
                if context.state.pageCount > 0 {
                    Text("P\(context.state.page)").font(.caption2.monospacedDigit()).foregroundStyle(visionGreen)
                } else {
                    Image(systemName: "text.viewfinder").foregroundStyle(visionGreen)
                }
            } minimal: {
                Image(systemName: icon(context.state.status)).foregroundStyle(visionGold)
            }
            .widgetURL(URL(string: "frankenocr://recognize"))
            .keylineTint(visionGold)
        }
    }
}

private struct OCRLockView: View {
    let context: ActivityViewContext<FrankenOCRRunActivityAttributes>
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon(context.state.status))
                .font(.title2.bold()).foregroundStyle(visionGold)
                .frame(width: 44, height: 44).background(visionGold.opacity(0.12), in: Circle())
            VStack(alignment: .leading, spacing: 3) {
                Text(context.state.stage).font(.headline).lineLimit(1)
                Text(context.state.detail).font(.caption).foregroundStyle(.secondary).lineLimit(1)
                OCRUnitRail(state: context.state)
            }
            Spacer(minLength: 4)
            OCRElapsedView(context: context)
        }
        .padding(15)
    }
}

private struct OCRElapsedView: View {
    let context: ActivityViewContext<FrankenOCRRunActivityAttributes>

    @ViewBuilder
    var body: some View {
        if context.state.status == .complete
            || context.state.status == .cancelled
            || context.state.status == .failed
        {
            Text(Self.clock(context.state.elapsedSeconds))
                .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
        } else {
            Text(timerInterval: context.attributes.startedAt...Date.distantFuture, countsDown: false)
                .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
        }
    }

    private static func clock(_ seconds: Int) -> String {
        let value = max(0, seconds)
        return String(format: "%d:%02d", value / 60, value % 60)
    }
}

private struct OCRUnitRail: View {
    let state: FrankenOCRRunContentState
    var body: some View {
        if state.totalUnits > 0, !state.totalIsEstimated {
            ProgressView(value: Double(state.completedUnits), total: Double(state.totalUnits)).tint(visionGold)
        } else {
            HStack(spacing: 3) {
                ForEach(0..<8, id: \.self) { index in
                    Capsule().fill(index < Int(state.completedUnits % 9) ? visionGold : visionGreen.opacity(0.2)).frame(height: 3)
                }
            }
        }
    }
}

private func icon(_ status: FrankenOCRRunContentState.Status) -> String {
    switch status {
    case .preparing: "photo.badge.arrow.down"
    case .vision: "eye.circle"
    case .reading: "text.viewfinder"
    case .assembling: "doc.text.image"
    case .complete: "checkmark.seal.fill"
    case .cancelled: "stop.circle"
    case .failed: "exclamationmark.triangle.fill"
    }
}

@main
struct FrankenOCRWidgetBundle: WidgetBundle {
    var body: some Widget {
        FrankenOCRVisionWidget()
        FrankenOCRLiveActivity()
    }
}
