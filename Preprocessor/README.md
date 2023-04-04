# Preprocessor

preprocessor.cpp takes all the images in the original dataset and converts them into images of specified size.

```
Usage: ./preprocessor [OPTIONS...]

Options:
	--help:		Print this help message
	--license:	Print the license
	--version:	Print the version number
	--quiet:	Don't write to stdout
	--root-dir:	Directory containing the dataset
	--size:		Specify a size for output images
```

As a result, all the images:

* are single-channel png images
* share the same aspect ratio (1:1)
* follow a simple naming convention, {X} _ {Y} _ {Z}.png where X is one of { "normal", "bacteria", "virus" }, Y is one of { "train", "test" }, and Z is the unique identifier of the image

Following public domain libraries are used:

* [stb_image.h](https://github.com/nothings/stb/blob/master/stb_image.h) : To read images
* [stb_image_resize.h](https://github.com/nothings/stb/blob/master/stb_image_resize.h) : To resize images
* [stb_image_write.h](https://github.com/nothings/stb/blob/master/stb_image_write.h) : To save images to the disk

## License

```
Preprocessor: A tool to pre-process images in the pneumonia dataset
Copyright (C) 2023 saccharineboi

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
```
