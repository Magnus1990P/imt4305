function [FFT, rFFT, iFFT, mFFT, pFFT, rIMG] = splitThatImage( img )
	%Convert to double
	img = double( img );

	%Fourier transform of 2D image
	F = fftshift( fft2( img ) );

	FFT		= F;
	rFFT	= real( F );
	iFFT	= imag( F );
	mFFT	= abs(  F );
	pFFT	= exp( j * angle(F) );
	rIMG	= ( mFFT .* pFFT );

	clear F, img;
end
