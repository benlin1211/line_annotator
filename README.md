# Welcome to line annotator!

## Description
This tool is for structured line fringe center annotation. 

By annotating the images, one can obtain a pixel-precision centerline position with high quality.

```
python main.py

python main.py --image_root=../../dataset/teeth_qisda/imgs_test_dummy/0727-0933/ --annotation_root=../../dataset/teeth_qisda/imgs_test_dummy/0727-0933_UV-only/ --save_path=./segment_result_0933 

python main.py --image_root=../../dataset/teeth_qisda/imgs_test_dummy/WangJung/0727-0949-1 --annotation_root=../../dataset/teeth_qisda/imgs_test_dummy/WangJung/0727-0949-1_UV-only/ --save_path=./segment_result_0949-1

python main.py --image_root=../../dataset/teeth_qisda/imgs_test_dummy/Albert/0727-0933-rand/ --annotation_root=../../dataset/teeth_qisda/imgs_test_dummy/0727-0933-rand_UV-only/ --save_path=./segment_result_0933-rand

```

## TODO:
- Fix bug that the drawn line width will accidentally change when pressing u after pressing z/x.
    - Workaround: press e after pressing z/x.

## Usages
![settings](https://raw.githubusercontent.com/benlin1211/line_annotator/main/images/setting.jpg)

## 這些線有一些特性（ MUST-READ ）

**1. [重要] 鄰近的間隔距離幾乎相同，並且大致平行**

- 所以如果有空一塊沒有線，可能要放大看一下那邊是不是真的沒有線，還是只是很暗的線但忘記標。

**2. [重要] 連續表面的線要連續，不連續表面則要斷掉**

- 例如：同一顆牙齒上的線，牙齒牙齦表面交界的線，應該都要連起來，不同牙齒表面的線則要斷掉，不要讓他們侵門踏戶。
- 其他像是泡泡、陰影裡的線，除非是幾乎看不到線的，否則請盡量讓他們連起來。

**3. 我覺得應該用 default 的筆刷寬度就好了。**

- 除非真的是很粗的線再調大。
- 或是有些線看起來比較暗比較窄的，希望可以調小筆刷去標~

**4. 備註**
- 請大約標記7-10張之後傳給我檢查一次
_____________________________________
### 1. Annotation Tools
**l: Line mode (default)**

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


### 2. Display
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

### 3. Save and Leave  
**s: Save**

Press key 's' to save the annotated result.

The results will be saved at {args.save_path, "annotation"}, {args.save_path, "image"}, {args.save_path, "combined"} respectively.

**esc: Leave without saving**

Press key esc to exit the program.

### 4. File I/O
**m: Load image**

Press key 'm' to load next image from {args.image_root}.

**n: Load annotation**

Press key 'a' to load existing annotation from {args.save_path, "annotation"}.


