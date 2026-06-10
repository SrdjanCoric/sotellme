(function () {
  'use strict';

  var motionOK = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (motionOK) document.documentElement.classList.add('motion-ok');

  var reveals = document.querySelectorAll('.reveal');
  if (motionOK && reveals.length && 'IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          io.unobserve(entry.target);
        }
      });
    }, { threshold: 0.15 });
    reveals.forEach(function (el) { io.observe(el); });
  } else {
    reveals.forEach(function (el) { el.classList.add('is-visible'); });
  }

  var typed = document.getElementById('typed-question');
  if (typed && motionOK) {
    var fullText = typed.textContent;
    typed.textContent = '';
    var caret = document.createElement('span');
    caret.className = 'hq-caret';
    typed.appendChild(caret);
    var i = 0;
    setTimeout(function tick() {
      if (i < fullText.length) {
        typed.insertBefore(document.createTextNode(fullText.charAt(i)), caret);
        i += 1;
        setTimeout(tick, 22);
      } else {
        caret.remove();
      }
    }, 2400);
  }

  var diagram = document.getElementById('diagram');
  if (!diagram) return;

  var svg = document.getElementById('wires');
  var sessionNode = document.getElementById('session-node');
  var startBtn = document.getElementById('start-btn');
  var hint = document.getElementById('sn-hint');
  var count = document.getElementById('sn-count');

  var cvCard = document.getElementById('card-cv');
  var cvStatus = document.getElementById('cv-status');
  var cvInput = document.getElementById('cv-input');
  var cvReject = document.getElementById('cv-reject');
  var cvFileRow = document.getElementById('cv-file-row');
  var cvName = document.getElementById('cv-name');
  var cvSize = document.getElementById('cv-size');
  var cvText = document.getElementById('cv-text');
  var cvPasteToggle = document.getElementById('cv-paste-toggle');
  var cvReplaceLabel = document.getElementById('cv-replace-label');

  var targetCard = document.getElementById('card-target');
  var targetStatus = document.getElementById('target-status');
  var company = document.getElementById('company');
  var role = document.getElementById('role');

  var jdCard = document.getElementById('card-jd');
  var jdStatus = document.getElementById('jd-status');
  var jdText = document.getElementById('jd-text');
  var jdLink = document.getElementById('jd-link');
  var jdReject = document.getElementById('jd-reject');

  var demo = document.getElementById('demo-armed');

  var cvPasteMode = false;
  var cvHasFile = true;

  function cvConnected() {
    if (cvPasteMode) return cvText.value.trim().length >= 40;
    return cvHasFile;
  }

  function targetConnected() {
    return company.value.trim().length > 0 && role.value.trim().length > 0;
  }

  function jdLinkValid() {
    var v = jdLink.value.trim();
    if (!v) return false;
    if (/linkedin\.com/i.test(v)) return false;
    return /^https?:\/\/\S+\.\S+/.test(v);
  }

  function jdConnected() {
    if (demo.checked) return true;
    return jdText.value.trim().length >= 40 || jdLinkValid();
  }

  function setCardState(card, statusEl, connected) {
    card.classList.toggle('is-connected', connected);
    card.classList.toggle('is-pending', !connected);
    statusEl.textContent = connected ? 'Connected' : 'Pending';
  }

  function anchorFor(rect, contRect, side) {
    if (side === 'right') return { x: rect.right - contRect.left, y: rect.top + rect.height / 2 - contRect.top };
    if (side === 'left') return { x: rect.left - contRect.left, y: rect.top + rect.height / 2 - contRect.top };
    if (side === 'bottom') return { x: rect.left + rect.width / 2 - contRect.left, y: rect.bottom - contRect.top };
    return { x: rect.left + rect.width / 2 - contRect.left, y: rect.top - contRect.top };
  }

  function makePath(d, cls, normalized) {
    var p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    p.setAttribute('d', d);
    p.setAttribute('class', cls);
    if (normalized) p.setAttribute('pathLength', '1');
    return p;
  }

  function drawWires() {
    var contRect = diagram.getBoundingClientRect();
    svg.setAttribute('viewBox', '0 0 ' + contRect.width + ' ' + contRect.height);
    svg.innerHTML = '';

    var sessionRect = sessionNode.getBoundingClientRect();
    var convergeOffsets = [-28, 0, 28];

    [cvCard, targetCard, jdCard].forEach(function (card, idx) {
      var cardRect = card.getBoundingClientRect();
      var horizontal = sessionRect.left > cardRect.right + 24;
      var from = anchorFor(cardRect, contRect, horizontal ? 'right' : 'bottom');
      var to = anchorFor(sessionRect, contRect, horizontal ? 'left' : 'top');
      if (!horizontal) to.x += convergeOffsets[idx];

      var d;
      if (horizontal) {
        var midX = (from.x + to.x) / 2;
        d = 'M ' + from.x + ' ' + from.y + ' C ' + midX + ' ' + from.y + ', ' + midX + ' ' + to.y + ', ' + to.x + ' ' + to.y;
      } else {
        var midY = (from.y + to.y) / 2;
        d = 'M ' + from.x + ' ' + from.y + ' C ' + from.x + ' ' + midY + ', ' + to.x + ' ' + midY + ', ' + to.x + ' ' + to.y;
      }

      var connected = card.classList.contains('is-connected');
      svg.appendChild(makePath(d, connected ? 'wire wire--live' : 'wire wire--pending'));
      if (connected && motionOK) svg.appendChild(makePath(d, 'nd-pulse', true));
    });
  }

  function update() {
    var cv = cvConnected();
    var target = targetConnected();
    var jd = jdConnected();

    setCardState(cvCard, cvStatus, cv);
    setCardState(targetCard, targetStatus, target);
    setCardState(jdCard, jdStatus, jd);

    var n = [cv, target, jd].filter(Boolean).length;
    var armed = n === 3;

    count.textContent = n + ' / 3 connected';
    sessionNode.classList.toggle('is-armed', armed);
    startBtn.disabled = !armed;

    if (armed) {
      hint.textContent = 'All three inputs connected.';
    } else if (cv && target && !jd) {
      hint.textContent = 'Connect the job description to arm the start.';
    } else {
      hint.textContent = 'Connect all three inputs to arm the start.';
    }

    drawWires();
  }

  cvInput.addEventListener('change', function () {
    var file = cvInput.files && cvInput.files[0];
    if (!file) return;
    if (/\.docx$/i.test(file.name)) {
      cvReject.hidden = false;
      cvInput.value = '';
    } else {
      cvReject.hidden = true;
      cvHasFile = true;
      cvName.textContent = file.name;
      cvSize.textContent = (file.size / (1024 * 1024)).toFixed(1) + ' MB';
    }
    update();
  });

  cvPasteToggle.addEventListener('click', function () {
    cvPasteMode = !cvPasteMode;
    cvText.hidden = !cvPasteMode;
    cvFileRow.hidden = cvPasteMode;
    cvReplaceLabel.hidden = cvPasteMode;
    cvReject.hidden = true;
    cvPasteToggle.textContent = cvPasteMode ? 'Use the file instead' : 'Paste text instead';
    if (cvPasteMode) cvText.focus();
    update();
  });

  [cvText, company, role, jdText].forEach(function (el) {
    el.addEventListener('input', update);
  });

  jdLink.addEventListener('input', function () {
    jdReject.hidden = !/linkedin\.com/i.test(jdLink.value);
    update();
  });

  demo.addEventListener('change', update);

  var resizeRaf = null;
  window.addEventListener('resize', function () {
    if (resizeRaf) cancelAnimationFrame(resizeRaf);
    resizeRaf = requestAnimationFrame(drawWires);
  });

  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(drawWires);
  }
  window.addEventListener('load', drawWires);

  update();
})();
