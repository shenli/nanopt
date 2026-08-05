window.MathJax = {
  tex: {
    // Arithmatex's generic mode emits these delimiters into the generated HTML.
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
  },
  options: {
    // Typeset only expressions wrapped by Arithmatex, not dollar signs in prose or code.
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

document$.subscribe(() => {
  MathJax.typesetPromise();
});
