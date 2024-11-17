to train the model you have to make a dataset for it the structue should be like this: <br> <br>
---<b>[root}</b><br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-----dataset<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---- right <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---- left <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---- straight <br>

in other words you should have a folder in the root directory named 'dataset' with 'right', 'left' and 'straight' subfolders in it with each of these folder<b> containtaining 80</b>, as the model was made to excel with 80 images more than that and it will overfit, if you you want to add more samples feel free to to tune the model. images</b>
