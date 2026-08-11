// Site chrome: glass header on scroll, scroll reveals, and nav section
// highlighting. Deliberately tiny and dependency-free — the page must be fully
// readable if this file never runs, so every effect here is additive.

// Mark the document as JS-capable BEFORE the first paint of the reveals: the
// stylesheet only hides .reveal under html.js, so with JS off (or blocked)
// everything is visible at rest instead of permanently transparent.
document.documentElement.classList.add("js");

const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ---------------------------------------------------------------- header */
const header = document.getElementById("site-header");
if (header) {
  let ticking = false;
  const apply = () => {
    ticking = false;
    header.classList.toggle("stuck", window.scrollY > 18);
  };
  window.addEventListener(
    "scroll",
    () => {
      if (!ticking) {
        ticking = true;
        requestAnimationFrame(apply);
      }
    },
    { passive: true },
  );
  apply();
}

/* --------------------------------------------------------------- reveals */
const revealables = [...document.querySelectorAll(".reveal")];

if (reduced) {
  for (const el of revealables) el.classList.add("revealed");
} else {
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) {
          entry.target.classList.add("revealed");
          observer.unobserve(entry.target);
        }
      }
    },
    { threshold: 0.06, rootMargin: "0px 0px -40px 0px" },
  );
  for (const el of revealables) observer.observe(el);

  // Instant jumps (scrollbar drags, the End key, a #deep-link processed before
  // this script ran) teleport past elements without ever firing an intersection,
  // which would leave them stuck at opacity 0. Sweep anything at or above the
  // fold on every scroll, and once at load.
  let queued = false;
  const sweep = () => {
    queued = false;
    for (const el of revealables) {
      if (
        !el.classList.contains("revealed") &&
        el.getBoundingClientRect().top < window.innerHeight + 220
      ) {
        el.classList.add("revealed");
        observer.unobserve(el);
      }
    }
  };
  window.addEventListener(
    "scroll",
    () => {
      if (!queued) {
        queued = true;
        requestAnimationFrame(sweep);
      }
    },
    { passive: true },
  );
  sweep();
  // A late-arriving webfont or image can reflow the page past a revealable
  // without a scroll event; one more sweep after load closes that window.
  window.addEventListener("load", sweep, { once: true });
}

/* ------------------------------------------------ drop zone, from a keyboard */
// The drop zone is a div that app.js turns into a file picker on click. Giving
// it role="button" and tabindex makes it reachable, but a div does not
// synthesize a click from Enter/Space the way a real <button> does — so a
// keyboard user would focus it and nothing would happen. This forwards the two
// keys the role promises; it adds no behaviour app.js does not already have.
const dropZone = document.getElementById("drop");
if (dropZone) {
  dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " " || e.key === "Spacebar") {
      e.preventDefault();
      dropZone.click();
    }
  });
}

/* ------------------------------------------------------- nav highlighting */
// aria-current on the link whose section owns the top third of the viewport.
const navLinks = [...document.querySelectorAll("#site-header nav a[href^='#']")];
const sections = navLinks
  .map((a) => ({ link: a, el: document.querySelector(a.getAttribute("href")) }))
  .filter((s) => s.el);

if (sections.length) {
  const spy = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        const hit = sections.find((s) => s.el === entry.target);
        if (!hit) continue;
        if (entry.isIntersecting) {
          for (const s of sections) s.link.removeAttribute("aria-current");
          hit.link.setAttribute("aria-current", "true");
        }
      }
    },
    { rootMargin: "-25% 0px -68% 0px" },
  );
  for (const s of sections) spy.observe(s.el);
}
