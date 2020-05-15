Part 1 : Simple C.
Part 2 : Regular CUDA code
Part 3 : CUDA with the 'shared' attribute
Part 4 : CUDA with Streams
Part 5 : CUDA with the 'shared' attribute + streams




DIM 1 :  
		dim3 dim1( 32, 32 );
        dim3 dim2(( cols - 1) / dim1.x + 1 , ( rows - 1 ) / dim1.y + 1 );

DIM 2 :  
		dim3 dim1( 16, 16 );
		dim3 dim2( 3 * 2 * (( cols - 1) / (dim1.x + 1) , 2 *( rows - 1 ) / (dim1.y) + 1 ));


IMG 1 = in.jpg    (512 x 512)
IMG 2 = in2.jpg	  (1920 x 1200)
