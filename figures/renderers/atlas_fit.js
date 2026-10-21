
function fitCompleteLogits() {
  document.querySelectorAll('.feature.dense .token-line').forEach(line => {
    const items = [...line.querySelectorAll('.token-item')];
    items.forEach(item => item.hidden = false);
    const rightEdge = line.getBoundingClientRect().right;
    let overflowed = false;
    items.forEach(item => {
      if (overflowed || item.getBoundingClientRect().right > rightEdge + 0.25) {
        item.hidden = true;
        overflowed = true;
      }
    });
  });
}
fitCompleteLogits();
