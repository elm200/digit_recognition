function argmax(x) {
  var i = 0;
  var idx;
  var max = -Infinity;
  x.tolist().forEach(function(n, i) {
    if(n > max) {
      idx = i;
      max = n;
    }
    i++;
  });
  return idx;
};

function predict(x) {
  var a1 = nj.dot(x, net_params.w1).add(net_params.b1);
  var z1 = nj.sigmoid(a1);
  var a2 = nj.dot(z1, net_params.w2).add(net_params.b2);
  var z2 = nj.sigmoid(a2);
  var a3 = nj.dot(z2, net_params.w3).add(net_params.b3);
  var z3 = nj.softmax(a3);
  return z3;
}

function clear() {
  var ctx = $('#canvas')[0].getContext('2d');
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, 224, 224);
}

function dup_array_like(array_like) {
  var list = new Array();
  var i;
  for (i = 0; i < array_like.length; i++) {
    list.push(array_like[i]);
  }
  return list;
}

function draw_probs(y) {
  var i;
  for (i = 0; i < 10; i++) {
    $("#prob" + i).text(i + ": " + (y.get(i) * 100).toString().slice(0, 4) + "%");
  }
}

function recognize() {
  var ctx = $('#canvas')[0].getContext("2d");
  var image = ctx.getImageData(0, 0, 224, 224).data;
  // it doesn't work
  // unless we copy image above to another array for some reason.
  var img = nj.array(dup_array_like(image)).reshape(224, 224, 4);
  img = nj.images.resize(img, 28, 28);
  img = nj.images.rgb2gray(img);
  img = nj.ones(img.shape).multiply(255).subtract(img);
  var transformed = document.getElementById('transformed');
  nj.images.save(img, transformed);
  // normalization and serialization
  var x = nj.divide(img, 255.0).reshape(28 * 28);
  var y= predict(x);
  draw_probs(y);
  var a = argmax(y);
  $('#result').text(a);
}

$(function() {
  $("#btn_recognize").click(function() {
    recognize();
  });

  $("#btn_clear").click(function() {
    clear();
  });

  $('#canvas').on("mousemove", function(e) {
    if(e.buttons === 1) {
      var ctx = $('#canvas')[0].getContext('2d');
      ctx.fillStyle = 'black';
      ctx.fillRect(e.offsetX, e.offsetY, 12, 12);
    }
  });

  clear();
});
