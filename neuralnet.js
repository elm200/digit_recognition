function argmax(x) {
  var i = 0;
  var idx = null;
  var max = -100000000000; // an arbitrary small number
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

$(function() {
  $("#btn").click(function() {
    var canvas = document.getElementById('canvas');
    var ctx = canvas.getContext("2d");
    var image = ctx.getImageData(0, 0, 224, 224).data;
    var list = new Array();
    var i;
    for (i = 0; i < image.length; i++) {
      list.push(image[i]);
    }
    var image2 = nj.array(list);
    var img = image2.reshape(224, 224, 4);
    img = nj.images.resize(img, 28, 28);
    img = nj.images.rgb2gray(img);
    img = nj.ones(img.shape).multiply(255).subtract(img);
    //
    var transformed = document.getElementById('transformed');
    nj.images.save(img, transformed);

    var img2 = nj.divide(img, 255.0);
    var img3 = img2.reshape(28 * 28);
    console.log(img3.shape);
    var a = argmax(predict(img3));
    console.log(a);
    // $('#result').text(a);
  });

  $("#clear").click(function() {
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



