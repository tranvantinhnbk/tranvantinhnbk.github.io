// The source code from: https://medium.com/@tarik.nzl/creating-a-canvas-component-with-free-hand-drawing-with-rxjs-and-angular-61279f577415
import { CommonModule } from '@angular/common';
import {
  Component,
  Input,
  ElementRef,
  AfterViewInit,
  ViewChild
} from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { fromEvent } from 'rxjs';
import { switchMap, takeUntil, pairwise } from 'rxjs/operators';
import * as tf from '@tensorflow/tfjs';


@Component({
  selector: 'app-canvas',
  templateUrl: './canvas.component.html',
  styleUrls: ['./canvas.component.scss'],
  standalone: true,
  imports: [MatButtonModule, CommonModule],
})
export class CanvasComponent implements AfterViewInit {
  @ViewChild('canvas') public canvas: ElementRef | undefined;

  @Input() public width = 400;
  @Input() public height = 400;

  private cx?: CanvasRenderingContext2D | null;
  capturedImageDataUrl: string | null = null;
  normalizeImage: any | null = null;
  dataUrl: string = '';

  imageAfterPreProcess?: number[][];
  predictNumber?: number;

  public ngAfterViewInit() {
    const canvasEl: HTMLCanvasElement = this.canvas?.nativeElement;
    
    this.cx = canvasEl.getContext('2d');

    canvasEl.width = this.width;
    canvasEl.height = this.height;

    if (!this.cx) throw 'Cannot get context';

    this.cx.lineWidth = 28;
    this.cx.lineCap = 'round';
    this.cx.strokeStyle = 'rgb(0,0,0)';

    this.captureEvents(canvasEl);
  }

  private captureEvents(canvasEl: HTMLCanvasElement) {
    // this will capture all mousedown events from the canvas element
    fromEvent(canvasEl, 'mousedown')
      .pipe(
        switchMap(e => {
          // after a mouse down, we'll record all mouse moves
          return fromEvent(canvasEl, 'mousemove').pipe(
            // we'll stop (and unsubscribe) once the user releases the mouse
            // this will trigger a 'mouseup' event
            takeUntil(fromEvent(canvasEl, 'mouseup')),
            // we'll also stop (and unsubscribe) once the mouse leaves the canvas (mouseleave event)
            takeUntil(fromEvent(canvasEl, 'mouseleave')),
            // pairwise lets us get the previous value to draw a line from
            // the previous point to the current point
            pairwise()
          );
        })
      )
      .subscribe((res) => {
        const rect = canvasEl.getBoundingClientRect();
        const prevMouseEvent = res[0] as MouseEvent;
        const currMouseEvent = res[1] as MouseEvent;

        // previous and current position with the offset
        const prevPos = {
          x: prevMouseEvent.clientX - rect.left,
          y: prevMouseEvent.clientY - rect.top
        };

        const currentPos = {
          x: currMouseEvent.clientX - rect.left,
          y: currMouseEvent.clientY - rect.top
        };

        // this method we'll implement soon to do the actual drawing
        this.drawOnCanvas(prevPos, currentPos);
      });
  }

  private drawOnCanvas(
    prevPos: { x: number; y: number },
    currentPos: { x: number; y: number }
  ) {
    if (!this.cx) {
      return;
    }

    this.cx.beginPath();

    if (prevPos) {
      this.cx.moveTo(prevPos.x, prevPos.y); // from
      this.cx.lineTo(currentPos.x, currentPos.y);
      this.cx.stroke();
    }
  }

  public clearCanvas() {
    const canvasEl: HTMLCanvasElement = this.canvas?.nativeElement;
    this.cx?.clearRect(0, 0, canvasEl.width, canvasEl.height);

  }

  findCorners(image: number[][]): number[] {
    let topLeftX = 0;
    let topLeftY = 0;
    let bottomRightX = 0;
    let bottomRightY = 0;
    const rows = image.length;
    const columns = image[0].length;
  
    // Find top-left Y
    for (let y = 0; y < rows; y++) {
      let check = true;
      for (let x = 0; x < columns; x++) {
        if (image[y][x] > 0) {
          check = false;
          break;
        }
      }
      if (!check) {
        topLeftY = y;
        break;
      }
    }
  
    // Find top-left X
    for (let x = 0; x < columns; x++) {
      let check = true;
      for (let y = 0; y < rows; y++) {
        if (image[y][x] > 0) {
          check = false;
          break;
        }
      }
      if (!check) {
        topLeftX = x;
        break;
      }
    }
  
    // Find bottom-right X
    for (let x = columns - 1; x >= 0; x--) {
      let check = true;
      for (let y = 0; y < rows; y++) {
        if (image[y][x] > 0) {
          check = false;
          break;
        }
      }
      if (!check) {
        bottomRightX = x;
        break;
      }
    }
  
    // Find bottom-right Y
    for (let y = rows - 1; y >= 0; y--) {
      let check = true;
      for (let x = 0; x < columns; x++) {
        if (image[y][x] > 0) {
          check = false;
          break;
        }
      }
      if (!check) {
        bottomRightY = y;
        break;
      }
    }
  
    return [topLeftX, topLeftY, bottomRightX, bottomRightY];
  }


  async preProcessImage() {
    if (this.cx && this.canvas) {
      let image: ImageData = this.cx.getImageData(0, 0, this.width, this.height);
      let imageData: Uint8ClampedArray = image.data;
      // Extract the black channel (assuming it's the first channel in the RGB format)
      let x = tf.tensor(Array.prototype.slice.call(imageData),[400,400,4], 'float32');// get the tensor 2d image with 4 channel
      let x_GrayScale = tf.gather(x, [3], 2) // just take red color
      x_GrayScale = x_GrayScale.reshape([400,400,1])

      const [topLeftX, topLeftY, bottomRightX, bottomRightY] = this.findCorners(x_GrayScale.arraySync() as number[][]);
      let column = bottomRightX - topLeftX+1;
      let row = bottomRightY - topLeftY+1;
      let edge = Math.max(column, row);

      // Testing bounding box
      // this.cx.strokeStyle = 'red';
      // this.cx.lineWidth = 2;
      // this.cx.beginPath();
      // this.cx.rect(topLeftX, topLeftY, column, row);
      // this.cx.stroke();

      //Get image with bounding box
      let cropImage = this.cx.getImageData(topLeftX, topLeftY, edge, edge);
      x = tf.tensor(Array.prototype.slice.call(cropImage.data),[edge,edge,4]);
      x = tf.div(x, 255);
      x_GrayScale = tf.gather(x, [3], 2)// just take red color
      x_GrayScale = tf.image.resizeNearestNeighbor(x_GrayScale as tf.Tensor3D, [20,20]);
      x_GrayScale = x_GrayScale.reshape([1,20,20,1])

      const Center = this.Center_Mass(x_GrayScale.reshape([20,20]).arraySync())
      const dX = 14-Center[0]
      const dY = 14-Center[1]

      let img = x_GrayScale.reshape([20,20]).arraySync() as number[][]

      let matrix = Array.from({ length: 28 }, () => Array(28).fill(0));
      for (let y = 0; y < 20; y++) {
        for (let x = 0; x < 20; x++) {
          if(y+dY <28 && x+dX < 28)
          {
            matrix[y+dY][x+dX] = img[y][x]
          }        
        }
      }
      this.imageAfterPreProcess = matrix;
      this.dataUrl = visualizeImage(matrix)
    }
  }

  async makePrediction() {
      await this.preProcessImage();
      let matrixTensor = tf.tensor(this.imageAfterPreProcess!.flat())
      matrixTensor = matrixTensor.reshape([1,28,28,1])

      const model = await tf.loadLayersModel('/assets/model.json')
      const output =  (model.predict(matrixTensor) as tf.Tensor<tf.Rank>).reshape([10]).argMax().dataSync()
      this.predictNumber = output[0];
  }

  Center_Mass(image: any) //the function get coordinate of Center of image inspire by Physic "Center of Mass"
  {
    const rows = image.length;
    const columns = image[0].length;
    let Center_X = 0;
    let Center_Y = 0;
    let Mass = 0;

    for (let y = 0; y < columns; y++) {
      for (let x = 0; x < rows; x++) {
        Center_X += x * image[y][x];
        Center_Y += y * image[y][x];
        Mass += image[y][x];
      }
    }
    Center_X /= Mass;
    Center_Y /= Mass;
    return  [Math.round(Center_X),Math.round(Center_Y)]
  }
}

function visualizeImage(matrix: number[][]) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');

  const width = matrix[0].length;
  const height = matrix.length;

  const borderWidth = 5; // Set the border width
  const borderColor = 'black'; // Set the border color

  canvas.width = width + borderWidth * 2; // Add border width to the canvas width
  canvas.height = height + borderWidth * 2; // Add border width to the canvas height

  // Draw the border
  ctx!.strokeStyle = borderColor;
  ctx!.lineWidth = borderWidth;
  ctx!.strokeRect(borderWidth / 2, borderWidth / 2, width + borderWidth, height + borderWidth);

  // Draw the image
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const pixelValue = matrix[y][x] * 255;
      const color = `rgb(${pixelValue}, ${pixelValue}, ${pixelValue})`;
      ctx!.fillStyle = color;
      ctx!.fillRect(x + borderWidth, y + borderWidth, 1, 1); // Add border width to the coordinates
    }
  }
  return canvas.toDataURL(); // Convert canvas to data URL
}
