import numpy as np
import matplotlib.pyplot as plt
from FSCUtil import FSCutil
import smallestenclosingcircle


class ManRes:


	halfMap1 = [];
	halfMap2 = [];
	apix = 0.0;
	frequencyMap = [];
	FSCdata = [];
	resVec = [];
	qVals = [];
	resolution = 0.0;
	embeddings = [];
	sizeMap = 0;


	#---------------------------------------------
	def ManRes(self, embeddingData, size):

		np.random.seed(3);
		self.embeddings = embeddingData;
		self.sizeMap = size;
		self.make_half_maps();
		self.standardizeResolution();
		self.calculate_frequency_map();
		maskData = np.ones(self.halfMap1.shape);

		tmpResVec, FSC, _, _, qVals_FDR, resolution_FDR, _ = FSCutil.FSC(self.halfMap1, self.halfMap2, maskData, self.apix, 0.143, 1, False, False, None);

		self.resolution = resolution_FDR;
		self.FSCdata = FSC;
		self.qVals = qVals_FDR;
		self.resVec = tmpResVec;

		self.writeFSC();

		print(self.resolution);


	#---------------------------------------------
	def make_half_maps(self):

		#initialize the half maps
		tmpHalfMap1 = np.zeros((self.sizeMap, self.sizeMap));
		tmpHalfMap2 = np.zeros((self.sizeMap, self.sizeMap));


		numLocalizations = self.embeddings.shape[0];
		sizeHalfSet = int(numLocalizations/2);


		#make the grid
		minX = np.amin(self.embeddings[:,0]);
		maxX = np.amax(self.embeddings[:,0]);
		minY = np.amin(self.embeddings[:,1]);
		maxY = np.amax(self.embeddings[:,1]);


		spacingX = (maxX-minX)/float(self.sizeMap -1);
		spacingY = (maxY-minY)/float(self.sizeMap -1);

		spacing = max(spacingX, spacingY);
		self.apix = spacing;

		#split the localizations randomly in 2 half sets
		permutedSequence = np.random.permutation(np.arange(numLocalizations));
		half1 = self.embeddings[permutedSequence[0:sizeHalfSet], :];
		half2 = self.embeddings[permutedSequence[sizeHalfSet:], :];

		print("make halfmap 1 ...");
		#place localizations of HalfSet1
		for i in range(half1.shape[0]):

			#transform localization to the grid
			indicesInGrid = np.floor((half1[i, :] - np.array([minX, minY]))/spacing);
			indicesInGrid = indicesInGrid.astype(int);
			tmpHalfMap1[indicesInGrid[0], indicesInGrid[1]] = tmpHalfMap1[indicesInGrid[0], indicesInGrid[1]] + 1.0;

		print("make halfmap 2 ...");
		#place localizations of HalfSet2
		for i in range(half2.shape[0]):

			#transform localization to the grid
			indicesInGrid = np.floor((half2[i, :] - np.array([minX, minY]))/spacing);
			indicesInGrid = indicesInGrid.astype(int);
			tmpHalfMap2[indicesInGrid[0], indicesInGrid[1]] = tmpHalfMap2[indicesInGrid[0], indicesInGrid[1]] + 1.0;

		self.halfMap1 = tmpHalfMap1;
		self.halfMap2 = tmpHalfMap2;


	#---------------------------------------------
	def calculate_frequency_map(self):

		#*********************************************************
		#*** calculation of the frequency map of the given map ***
		#*********************************************************

		sizeMap = self.halfMap1.shape;

		if self.halfMap1.ndim == 3:
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

		elif self.halfMap1.ndim == 2:
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


	#---------------------------------------------
	def standardizeResolution(self):

		#calculate convex hull
		print("Calculate smalles enclosing circle ...");
		hull = smallestenclosingcircle.make_circle(self.embeddings);

		print(hull[2]);

		"""numVertices = hull.points.shape[0];
		#now get maximum distance between two vertices
		maxDist = 0.0;
		for ind1 in range(numVertices-1):
			for ind2 in range(ind1+1, numVertices):

				tmpDist = np.sqrt(np.sum((hull.points[ind1,:] - hull.points[ind2,:])**2));

				if tmpDist > maxDist:
					maxDist = tmpDist;"""


		self.apix = self.apix/hull[2];


