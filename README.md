# Welcome to line annotator!

## Description
This tool is for structured line fringe center annotation. 

By annotating the images, one can obtain a pixel-precision centerline position with high quality.

## Usages

### Annotation Tools
- **l: Line mode (default)**

Press 'l' to enable line mode. 

Under line mode, the cursor will automatically search and connect to the nearest endpoint within thw brush range while drawing lines. 

**e: Eraser mode**

Press key 'e' to enable eraser mode. 

Under eraser mode, the annotation within the brush range will be erased when clicking. (Note: this function cannot undo/redo.)

**u: Undo**

Press key 'u' to undo last line painted.

**r: Redo**

Press key 'r' to redo last line undone.

**z: Decrease brush size**

Press key 'z' to decrease brush size of current mode by 1.

**x: Increase brush size**

Press key 'x' to increase brush size of current mode by 1.


### Display
**a: Display background only**

Press key 'a' to display background only. 

Press 'a' again to leave display mode.

Note: Under this mode, the other keys and brush are disabled, but you can scroll to zoom-in/out, and drag to move for a better FOV of annotation.  

**b: Hide background**

Press key 'b' to hide background. 

Press 'b' again to show background.

**p: Show all endpoints. (default)**

Press key 'p' to hide endpoints. 

Press 'p' again to show endpoints.

**h: Help**

Press key 'h' to print help on console.

### Save and Leave  
**s: Save**

Press key 's' to save the annotated result.

The results will be saved at {args.save_path, "annotation"}, {args.save_path, "image"}, {args.save_path, "combined"} respectively.

**esc: Leave without saving**

Press key esc to exit the program.

### File I/O
**m: Load image**

Press key 'm' to load next image from {args.image_root}.

**n: Load annotation**

Press key 'a' to load existing annotation from {args.save_path, "annotation"}.


