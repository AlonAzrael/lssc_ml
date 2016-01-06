function getRandomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function refresh_canvas (canvas) {
  var ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height )
  ctx.fillStyle = "rgb(255,255,255)";
  ctx.fillRect(0, 0, canvas.width, canvas.height)
}

function setup_canvas (canvas_id) {
  var main_canvas = document.getElementById(canvas_id);
  var ctx = main_canvas.getContext('2d');

  ctx.lineWidth = 1;
  ctx.lineJoin = ctx.lineCap = 'round';
  ctx.strokeStyle = 'black';

  var isDrawing, lastPoint;

  var mouse = {
    get_x: function (evt) {
      var offset = $('#'+canvas_id).offset();
      var x = evt.pageX - offset.left;
      return x
    },
    get_y: function (evt) {
      var offset = $('#'+canvas_id).offset();
      var y = evt.pageY - offset.top;
      return y
    },
  }

  main_canvas.onmousedown = function(evt) {
    isDrawing = true;
    lastPoint = { x: mouse.get_x(evt), y: mouse.get_y(evt) }
  };

  main_canvas.onmousemove = function(evt) {
    if (!isDrawing) return;

    ctx.beginPath();
    
    ctx.moveTo(lastPoint.x - getRandomInt(0, 2), lastPoint.y - getRandomInt(0, 2));
    ctx.lineTo(mouse.get_x(evt) - getRandomInt(0, 2), mouse.get_y(evt) - getRandomInt(0, 2));
    ctx.stroke();
    
    ctx.moveTo(lastPoint.x, lastPoint.y);
    ctx.lineTo(mouse.get_x(evt), mouse.get_y(evt));
    ctx.stroke();
    
    ctx.moveTo(lastPoint.x + getRandomInt(0, 2), lastPoint.y + getRandomInt(0, 2));
    ctx.lineTo(mouse.get_x(evt) + getRandomInt(0, 2), mouse.get_y(evt) + getRandomInt(0, 2));
    ctx.stroke();
      
    lastPoint = { x: mouse.get_x(evt), y: mouse.get_y(evt) };
  };

  main_canvas.onmouseup = function() {
    isDrawing = false;
  };

  return main_canvas
}

