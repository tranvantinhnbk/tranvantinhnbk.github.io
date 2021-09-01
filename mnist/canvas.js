
window.addEventListener('load', () => {
  Program()
});

function Program()
{
  let canvas = document.getElementById("canvas");
  let  ctx = canvas.getContext("2d");
  ctx.fillStyle = 'rgb(255,255,255)'
  ctx.fillRect(0,0,canvas.width, canvas.height)


  let reset = document.getElementById("reset");
  let predict = document.getElementById("predict");
  let painting = 0
  let check = 0

  function startPosition(e)
  {
    painting = true;
    draw(e);
  }
  function finishedPosition()
  {
    painting = false;
    ctx.beginPath();
  }
  function draw(e)
  {
      if(!painting) return;
      ctx.lineWidth = 28;
      ctx.lineCap = "round";
      ctx.lineTo(e.offsetX, e.offsetY);
      ctx.strokeStyle = 'rgb(0,0,0)';
      ctx.stroke();
  }
  function resetbutton()
  {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'rgb(255,255,255)'
    ctx.fillRect(0,0,canvas.width, canvas.height)
  }
  async function processImageData()
  {

    image = ctx.getImageData(0,0,280,280);
    imageData = image.data;
    x = tf.tensor(Array.prototype.slice.call(imageData),[280,280,4], 'float32');// get the tensor 2d image with 4 channel
    x = tf.div(tf.sub(tf.scalar(255), x), 255); // normalize the matrix to 0_1
    x_GrayScale = tf.gather(x, [0], axis = 2) // just take red color
    x_GrayScale = x_GrayScale.reshape([280,280,1])



    locate = getImage(x_GrayScale.arraySync())

    row = locate[2]-locate[0]+1
    column = locate[3]-locate[1]+1

    // initial row = collumn to get square contain image
    if(row>column)
      column = row
    else
      row = column

    image = ctx.getImageData(locate[0],locate[1],row,column);
    imageData = image.data;
    x = tf.tensor(Array.prototype.slice.call(imageData),[row,column,4], 'float32');
    x = tf.div(tf.sub(tf.scalar(255), x), 255);
    x_GrayScale = tf.gather(x, [0], axis = 2)
    x_GrayScale = tf.image.resizeNearestNeighbor(x_GrayScale, [20,20])
    x_GrayScale = x_GrayScale.reshape([1,20,20,1])

    Center = Center_Mass(x_GrayScale.reshape([20,20]).arraySync())
    dX = 14-Center[0]
    dY = 14-Center[1]


    image = x_GrayScale.reshape([20,20]).arraySync()


    matrix = tf.zeros([28,28])
    matrix = matrix.arraySync()
    // the loop try to put the image into 28*28 canvas while center mass of digit is the center of 28*28 canvas( (14,14))
    for(let x = 0; x < 20; x++)
      for(let y = 0; y < 20; y++)
      {
        if(y+dY <28 && x+dX < 28)
        {
        matrix[y+dY][x+dX] = image[y][x]
        }
      }
    matrix = tf.tensor(matrix.flat())
    matrix = matrix.reshape([1,28,28,1])

    model = await tf.loadLayersModel('./model.json')
    result = document.getElementById("result")
    output =  model.predict(matrix).reshape([10]).argMax().dataSync()
    result.innerHTML = "<span style='font-size:200px'>"+ output +"</span>";

  }
  // EventListeners
  canvas.addEventListener("mousedown", startPosition);
  canvas.addEventListener("mouseup", finishedPosition);
  canvas.addEventListener("mousemove", draw)
  reset.addEventListener("click", resetbutton)
  predict.addEventListener("click", processImageData)
}
function getImage(image) // the function use to get the digit in image by remove all row and columnn are nearly black // image is 2D js Array
{
  topleftX = 0;
  topleftY = 0;
  bottomrightX = 0;
  bottomrightY = 0;
  topleftY = 0;
  columns =image[0].length
  rows = image.length
  for(let y = 0; y < rows; y++)
    {
      check = true;
      for(let x = 0; x < columns; x++)
        {
          if(image[y][x] > 0)
          {
              check = false
          }
        }
      if(check == false)
      {
        topleftY = y;
        break;
      }

    }
  for(let x = 0; x < columns; x++)
      {
        check = true;
        for(let y = 0; y < rows; y++)
          {
            if(image[y][x] > 0)
            {
                check = false
            }
          }
        if(check == false)
        {
          topleftX = x;
          break;
        }
      }
  for(let x = rows-1; x >= 1; x--)
    {
      check = true;
        for(let y = 0; y < columns; y++)
         {
           if(image[y][x] > 0)
              check = false
         }
      if(check == false)
      {
        bottomrightX = x
        break;
      }
    }
    for(let y = columns-1; y >= 1; y--)
      {
        check = true;
          for(let x = 0; x < rows; x++)
           {
             if(image[y][x] > 0)
                check = false
           }
        if(check == false)
        {
          bottomrightY = y
          break;
        }
      }
    return [topleftX,topleftY,bottomrightX,bottomrightY] //TOPLEFT and BOTTOMRIGHT
}
function Center_Mass(image) //the function get coordinate of Center of image inspire by Physic "Center of Mass"
{
  rows = image.length
  columns = image[0].length
  Center_X = 0
  Center_Y = 0
  Mass = 0
  for(let y = 0; y < columns; y++ )
    for(let x = 0; x < rows; x++)
      {
        Center_X += x * image[y][x];
        Center_Y += y * image[y][x];
        Mass += image[y][x];
      }
  Center_X /= Mass
  Center_Y /= Mass
  return  [Math.round(Center_X),Math.round(Center_Y)]
}


// https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4
