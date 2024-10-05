const argmax = (x) => {
  let idx;
  let max = -Infinity;
  x.tolist().forEach((n, i) => {
    if (n > max) {
      idx = i;
      max = n;
    }
  });
  return idx;
};

const predict = (x) => {
  const a1 = nj.dot(x, window.net_params.w1).add(window.net_params.b1);
  const z1 = nj.sigmoid(a1);
  const a2 = nj.dot(z1, window.net_params.w2).add(window.net_params.b2);
  const z2 = nj.sigmoid(a2);
  const a3 = nj.dot(z2, window.net_params.w3).add(window.net_params.b3);
  const z3 = nj.softmax(a3);
  return z3;
};

const clear = () => {
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, 224, 224);
};

const dupArrayLike = (arrayLike) => Array.from(arrayLike);

const drawProbs = (y) => {
  for (let i = 0; i < 10; i++) {
    document.getElementById(`prob${i}`).textContent =
      `${i}: ${(y.get(i) * 100).toFixed(2)}%`;
  }
};

const recognize = () => {
  const canvas = document.getElementById("canvas");
  const ctx = canvas.getContext("2d");
  const image = ctx.getImageData(0, 0, 224, 224).data;
  const img = nj.array(dupArrayLike(image)).reshape(224, 224, 4);
  const resizedImg = nj.images.resize(img, 28, 28);
  const grayImg = nj.images.rgb2gray(resizedImg);
  const invertedImg = nj.ones(grayImg.shape).multiply(255).subtract(grayImg);
  const transformed = document.getElementById("transformed");
  nj.images.save(invertedImg, transformed);
  const x = nj.divide(invertedImg, 255.0).reshape(28 * 28);
  const y = predict(x);
  drawProbs(y);
  const a = argmax(y);
  document.getElementById("result").textContent = a;
};

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("btn_recognize").addEventListener("click", recognize);
  document.getElementById("btn_clear").addEventListener("click", clear);

  const canvas = document.getElementById("canvas");
  canvas.addEventListener("mousemove", (e) => {
    if (e.buttons === 1) {
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "black";
      ctx.fillRect(e.offsetX, e.offsetY, 12, 12);
    }
  });

  clear();
});
