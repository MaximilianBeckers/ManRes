import numpy as np
import pyfftw
import matplotlib.pyplot as plt
from FSCUtil import FSCutil

class ManRes:

	halfMap1 = [];
	halfMap2 = [];
	frequencyMap = [];
	FSCdata = [];
	resVec = [];
	qVals = [];
	resolution = 0.0;
	embeddings = [];
	sizeMap = 0;

	#---------------------------------------------
	def ManRes(self, embeddingData, size):

		self.embeddings = embeddingData;
		self.sizeMap = size;
		self.make_half_maps();
		self.calculate_frequency_map();

		maskData = np.ones(self.halfMap1.shape);
		resVec, FSC, _, _, qVals_FDR, resolution_FDR, _ = FSCutil.FSC(self.halfMap1, self.halfMap2, maskData, 1, 0.143, 1, False, True, None);

		self.resolution = resolution_FDR;
		self.FSCdata = FSC;
		self.qVals = qVals_FDR;

		self.writeFSC();

	#---------------------------------------------
	def make_half_maps(self):

		#initialize the half maps
		tmpHalfMap1 = np.zeros(self.sizeMap, self.sizeMap);
		tmpHalfMap2 = np.zeros(self.sizeMap, self.sizeMap);

		numLocalizations = self.embeddings.shape[0];
		sizeHalfSet = int(numLocalizations/2);


		#make the grid
		minX = np.amin(self.embeddings, 0);
		maxX = np.amax(self.embeddings, 0);
		minY = np.amin(self.embeddings, 1);
		maxY = np.amax(self.embeddings, 1);

		spacingX = (maxX-minX)/float(self.sizeMap);
		spacingY = (maxY-minY)/float(self.sizeMap);
		spacing = np.max(spacingX, spacingY);


		#split the localizations randomly in 2 half sets
		permutedSequence = np.random.permutation(np.arange(numLocalizations));
		half1 = self.embeddings[permutedSequence[0:sizeHalfSet], :];
		half2 = self.embeddings[permutedSequence[sizeHalfSet:], :];


		#place localizations of HalfSet1
		for i in range(half1.shape[0]):

			#transform localization to the grid
			indicesInGrid = int((self.embeddings[i, :] - np.array([minX, minY]))/spacing);
			tmpHalfMap1[indicesInGrid[0], indicesInGrid[1]] = tmpHalfMap1[indicesInGrid[0], indicesInGrid[1]] + 1.0;


		#place localizations of HalfSet2
		for i in range(half2.shape[0]):

			#transform localization to the grid
			indicesInGrid = int((self.embeddings[i, :] - np.array([minX, minY]))/spacing);
			tmpHalfMap2[indicesInGrid[0], indicesInGrid[1]] = tmpHalfMap2[indicesInGrid[0], indicesInGrid[1]] + 1.0;


		self.halfMap1 = tmpHalfMap1;
		self.halfMap2 = tmpHalfMap2;

	#---------------------------------------------
	def calculate_frequency_map(self):

		#*********************************************************
		#*** calculation of the frequency map of the given map ***
		#*********************************************************

		sizeMap = self.halfMap1.shape;

		if map.ndim == 3:
			# calc frequency for each voxel
			freqi = np.fft.fftfreq(sizeMap[0], 1.0);
			freqj = np.fft.fftfreq(sizeMap[1], 1.0);
			freqk = np.fft.rfftfreq(sizeMap[2], 1.0);

			sizeFFT = np.array([freqi.size, freqj.size, freqk.size]);
			FFT = np.zeros(sizeFFT);

			freqMapi = np.copy(FFT);
			for j in range(sizeFFT[1]):
				for k in range(sizeFFT[2]):
					freqMapi[:, j, k] = freqi * freqi;

			freqMapj = np.copy(FFT);
			for i in range(sizeFFT[0]):
				for k in range(sizeFFT[2]):
					freqMapj[i, :, k] = freqj * freqj;

			freqMapk = np.copy(FFT);
			for i in range(sizeFFT[0]):
				for j in range(sizeFFT[1]):
					freqMapk[i, j, :] = freqk * freqk;

			tmpFrequencyMap = np.sqrt(freqMapi + freqMapj + freqMapk);

		elif map.ndim == 2:
			# calc frequency for each voxel
			freqi = np.fft.fftfreq(sizeMap[0], 1.0);
			freqj = np.fft.fftfreq(sizeMap[1], 1.0);

			sizeFFT = np.array([freqi.size, freqj.size]);
			FFT = np.zeros(sizeFFT);

			freqMapi = np.copy(FFT);
			for j in range(sizeFFT[1]):
				freqMapi[:, j] = freqi * freqi;

			freqMapj = np.copy(FFT);
			for i in range(sizeFFT[0]):
				freqMapj[i, :] = freqj * freqj;

			tmpFrequencyMap = np.sqrt(freqMapi + freqMapj);

		self.frequencyMap = tmpFrequencyMap;

	# --------------------------------------------
	def writeFSC(self):

		# *******************************
		# ******* write FSC plots *******
		# *******************************

		plt.plot(self.resVec, self.FSCdata, label="FSC", linewidth=1.5);

		# threshold the adjusted pValues
		self.qVals[self.qVals <= 0.01] = 0.0;

		plt.plot(self.resVec[0:][self.qVals == 0.0], self.qVals[self.qVals == 0.0] - 0.05, 'xr',
				 label="sign. at 1% FDR");
		# plt.plot(resolutions[0:][pValues==0.0], pValues[pValues==0.0]-0.1, 'xb', label="sign. at 1%");

		plt.axhline(0.5, linewidth=0.5, color='r');
		plt.axhline(0.143, linewidth=0.5, color='r');
		plt.axhline(0.0, linewidth=0.5, color='b');

		plt.xlabel("1/resolution [1/A]");
		plt.ylabel("FSC");
		plt.legend();

		plt.savefig('FSC.png', dpi=300);
		plt.close();
