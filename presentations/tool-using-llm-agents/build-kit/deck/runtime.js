(function () {
  "use strict";
  var root = document.documentElement;
  var slides = Array.prototype.slice.call(document.querySelectorAll(".slide"));
  var N = slides.length;
  var idx = 0;
  var store = {
    get: function (k) { try { return window.localStorage.getItem(k); } catch (e) { return null; } },
    set: function (k, v) { try { window.localStorage.setItem(k, v); } catch (e) { /* storage unavailable */ } }
  };
  var $ = function (id) { return document.getElementById(id); };
  var mq = window.matchMedia("(min-width: 821px)");

  function setLang(lang) {
    root.setAttribute("data-lang", lang);
    root.setAttribute("lang", lang === "zh" ? "zh-CN" : "en");
    $("lang-en").setAttribute("aria-pressed", String(lang === "en"));
    $("lang-zh").setAttribute("aria-pressed", String(lang === "zh"));
    store.set("tula-lang", lang);
    if (slides[idx] && root.classList.contains("present")) fitSlide(slides[idx]);
    updateBar();
  }

  var themes = ["auto", "light", "dark"];
  function setTheme(th) {
    if (th === "auto") root.removeAttribute("data-theme"); else root.setAttribute("data-theme", th);
    var b = $("theme");
    b.setAttribute("data-mode", th);
    b.querySelector(".l-en").textContent = { auto: "Theme: auto", light: "Theme: light", dark: "Theme: dark" }[th];
    b.querySelector(".l-zh").textContent = { auto: "主题：自动", light: "主题：浅色", dark: "主题：深色" }[th];
    store.set("tula-theme", th);
  }

  function fit() {
    var present = mq.matches;
    root.classList.toggle("present", present);
    if (!present) {
      slides.forEach(function (el) { var b = el.querySelector(".s-fit"); if (b) b.style.removeProperty("--fit"); });
      return;
    }
    var vp = document.querySelector(".viewport");
    var w = vp.clientWidth, h = vp.clientHeight;
    var s = Math.min(w / 1280, h / 720) * 0.985;
    root.style.setProperty("--scale", String(Math.max(0.2, s)));
    if (slides[idx]) fitSlide(slides[idx]);
  }

  // Fit: scale each slide's content (CSS zoom, 0.85 to 1.22) so it fills the
  // body without overflowing. Computed per language and cached on the slide.
  var FIT_MIN = 0.85, FIT_MAX = 1.22;
  function curLang() { return root.getAttribute("data-lang") === "zh" ? "zh" : "en"; }
  function measureFit(el) {
    var body = el.querySelector(".s-body"), box = el.querySelector(".s-fit");
    if (!body || !box) return 1;
    var wide = Array.prototype.slice.call(el.querySelectorAll(".mblock, .tblwrap, pre.code"));
    var fits = function (k) {
      box.style.setProperty("--fit", String(k));
      if (box.getBoundingClientRect().height > body.getBoundingClientRect().height + 0.5) return false;
      for (var i = 0; i < wide.length; i++) if (wide[i].scrollWidth > wide[i].clientWidth + 1) return false;
      return true;
    };
    var lo, hi;
    if (fits(1)) { lo = 1; hi = FIT_MAX; if (fits(hi)) return hi; }
    else { lo = FIT_MIN; hi = 1; if (!fits(lo)) return lo; }
    for (var it = 0; it < 7; it++) { var mid = (lo + hi) / 2; if (fits(mid)) lo = mid; else hi = mid; }
    return Math.floor(lo * 1000) / 1000;
  }
  function fitSlide(el, force) {
    if (!root.classList.contains("present")) { el.querySelector(".s-fit") && el.querySelector(".s-fit").style.removeProperty("--fit"); return; }
    var key = "fit" + curLang();
    var shown = el.classList.contains("active");
    if (!shown) el.classList.add("active");
    var k = (!force && el.dataset[key]) ? parseFloat(el.dataset[key]) : measureFit(el);
    el.dataset[key] = String(k);
    el.querySelector(".s-fit").style.setProperty("--fit", String(k));
    if (!shown) el.classList.remove("active");
  }
  function fitAll(force) { slides.forEach(function (el) { fitSlide(el, force); }); }

  function updateBar() {
    var s = slides[idx];
    if (!s) return;
    $("count").textContent = (idx + 1) + " / " + N;
    var lang = root.getAttribute("data-lang") === "zh" ? "zh" : "en";
    $("sec").textContent = s.getAttribute("data-sec-" + lang) || "";
    document.querySelector(".progress").style.width = (100 * (idx + 1) / N) + "%";
  }

  function go(i, push) {
    idx = Math.max(0, Math.min(N - 1, i));
    slides.forEach(function (el, k) { el.classList.toggle("active", k === idx); });
    fitSlide(slides[idx]);
    updateBar();
    if (push !== false) {
      try { history.replaceState(null, "", "#s" + (idx + 1)); } catch (e) { /* ignore */ }
    }
  }

  function fromHash() {
    var m = /^#s(\d+)$/.exec(location.hash || "");
    return m ? parseInt(m[1], 10) - 1 : null;
  }

  function onKey(e) {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    var k = e.key;
    if (k === "ArrowRight" || k === "PageDown" || k === " " || k === "Enter") { go(idx + 1); e.preventDefault(); }
    else if (k === "ArrowLeft" || k === "PageUp" || k === "Backspace") { go(idx - 1); e.preventDefault(); }
    else if (k === "Home") { go(0); e.preventDefault(); }
    else if (k === "End") { go(N - 1); e.preventDefault(); }
    else if (k === "l" || k === "L") { setLang(root.getAttribute("data-lang") === "zh" ? "en" : "zh"); }
  }

  var tx = null;
  function onTouchStart(e) { if (e.touches.length === 1) tx = e.touches[0].clientX; }
  function onTouchEnd(e) {
    if (tx === null || !root.classList.contains("present")) return;
    var dx = e.changedTouches[0].clientX - tx;
    if (Math.abs(dx) > 50) go(idx + (dx < 0 ? 1 : -1));
    tx = null;
  }

  function start(data) {
    data = data || {};
    setTheme(themes.indexOf(store.get("tula-theme")) >= 0 ? store.get("tula-theme") : "auto");
    setLang(data.lang || store.get("tula-lang") || "en");
    fit();
    var h = fromHash();
    go(typeof data.idx === "number" ? data.idx : (h !== null ? h : 0), false);
    $("prev").addEventListener("click", function () { go(idx - 1); });
    $("next").addEventListener("click", function () { go(idx + 1); });
    $("lang-en").addEventListener("click", function () { setLang("en"); });
    $("lang-zh").addEventListener("click", function () { setLang("zh"); });
    $("theme").addEventListener("click", function () {
      var cur = $("theme").getAttribute("data-mode");
      setTheme(themes[(themes.indexOf(cur) + 1) % themes.length]);
    });
    document.addEventListener("keydown", onKey);
    window.addEventListener("resize", fit);
    if (mq.addEventListener) mq.addEventListener("change", fit);
    window.addEventListener("hashchange", function () { var h2 = fromHash(); if (h2 !== null) go(h2, false); });
    document.addEventListener("touchstart", onTouchStart, { passive: true });
    document.addEventListener("touchend", onTouchEnd, { passive: true });
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(function () { setTimeout(function () { fitAll(true); }, 30); });
    }
    window.deckAPI = { go: go, fitAll: fitAll, setLang: setLang };
    if (window.claude && window.claude.hot && window.claude.hot.snapshot) {
      window.claude.hot.snapshot(function () { return { idx: idx, lang: root.getAttribute("data-lang") }; });
    }
  }

  var hot = window.claude && window.claude.hot;
  if (hot && hot.ready) hot.ready(start); else start(hot && hot.data ? hot.data : {});
})();
