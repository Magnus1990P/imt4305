function main( fIMG )

	oIMG=[];
	if fIMG(end-2:end) == 'gif'
		display('gif');
		[c,map] = imread( fIMG );
		oIMG 		= rgb2gray(ind2rgb(c, map));
	elseif size(imread(fIMG),3)==3
		display('RGB');
		oIMG		= rgb2gray( imread(fIMG) );
	else
		display('Gray');
		oIMG		= imread(fIMG);
	end


	[FFT, rFFT, iFFT, mFFT, pFFT, rIMG] = splitThatImage( oIMG );


	subplot(2,3,1),  imshow( oIMG,[] ), 			title('Spat - Org');
	subplot(2,3,2),  imshow( ifft2(ifftshift(FFT)),[] ), 	title('Spat - Rec Org');
	subplot(2,3,3),  imshow( ifft2(ifftshift(rIMG)),[] ), 	title('Spat - Rec Org');


	subplot(2,3,4),  imshow( log(1+mFFT),[] ),	title('Freq - Magnitude');	
	subplot(2,3,5),  imshow( log(1+rFFT),[] ), 	title('Freq - Real');
	subplot(2,3,6),  imshow( log(1+iFFT),[] ), 	title('Freq - Imaginary');

	%subplot(2,3,5),  imshow( ifft2(ifftshift(Fp)),[] ), 				title('Phase - inverse');


end
