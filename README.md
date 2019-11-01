# Color Transfer between Images
<p align='center'>
  <img src='https://github.com/prasunroy/color-transfer/raw/master/assets/image_1.jpg' />
</p>

***An implementation of the paper ["Color Transfer between Images"](http://www.cs.northwestern.edu/~bgooch/PDFs/ColorTransfer.pdf) by Erik Reinhard, Michael Adhikhmin, Bruce Gooch and Peter Shirley (2001).***

![badge](https://github.com/prasunroy/color-transfer/blob/master/assets/badge_1.svg)
![badge](https://github.com/prasunroy/color-transfer/blob/master/assets/badge_2.svg)

## Installation
#### Step 1: Install dependencies
```
pip install numpy pillow
```
#### Step 2: Clone repository
```
git clone https://github.com/prasunroy/color-transfer.git
cd color-transfer
```

## Example
```python
import color_transfer as ct
image = ct.transfer_color(source_file='assets/sunset.jpg', target_file='assets/ocean.jpg', rescale=True)
image.save('assets/ocean_sunset.jpg')
image.show()
```

## License
MIT License

Copyright (c) 2018 Prasun Roy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

<br />
<br />

**Made with** :heart: **and GitHub**
