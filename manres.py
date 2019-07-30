import numpy as np
import scipy
import sys
import matplotlib.pyplot as plt
from Utilities import FSCutil,smallestEnclosingCircle, FDRutil

class ManRes:

	halfMap1 = [];
	halfMap2 = [];
	fullMap = [];
	filteredMap = [];
	apix = 0.0;
	frequencyMap = [];
	FSCdata = [];
	resVec = [];
	qVals = [];
	resolution = 0.0;
	embeddings = [];
	sizeMap = 0;
	encCircRad = 0.0;


	#---------------------------------------------
	def ManRes_halfSets(self, embeddingHalf1, embeddingHalf2, size):

		np.random.seed(3);
		
		self.embeddingsHalf1 = embeddingHalf1;
		self.embeddingsHalf2 = embeddingHalf2;

		#optimize procrustes distance
		#self.embeddingsHalf1, self.embeddingsHalf2, _ = scipy.spatial.procrustes(self.embeddingsHalf1, self.embeddingsHalf2);

		self.sizeMap = size;
		self.make_half_maps();
		self.fullMap = self.halfMap1 + self.halfMap2;
		self.standardizePixelSize();
		self.frequencyMap = FSCutil.calculate_frequency_map(self.halfMap1);
		maskData = np.ones(self.halfMap1.shape);

		tmpResVec, FSC, _, _, qVals_FDR, resolution_FDR, _ = FSCutil.FSC(self.halfMap1, self.halfMap2, maskData, self.apix, 0.143, 1, False, False, None);

		self.resolution = resolution_FDR;
		self.FSCdata = FSC;
		self.calcTTest();
		self.qVals = qVals_FDR;
		self.resVec = tmpResVec;
		self.filterMap();
		self.writeFSC();


	#---------------------------------------------
	def ManRes(self, embeddingData, size):

		np.random.seed(3);

		#split the localizations randomly in 2 half sets
		numLocalizations = embeddingData.shape[0];
		sizeHalfSet = int(numLocalizations/2);
		permutedSequence = np.random.permutation(np.arange(numLocalizations));
		self.embeddingsHalf1 = embeddingData[permutedSequence[0:sizeHalfSet], :];
		self.embeddingsHalf2 = embeddingData[permutedSequence[sizeHalfSet:], :];

		self.sizeMap = size;
		self.make_half_maps();
		self.fullMap = self.halfMap1 + self.halfMap2;
		self.standardizePixelSize();
		self.frequencyMap = FSCutil.calculate_frequency_map(self.halfMap1);
		maskData = np.ones(self.halfMap1.shape);

		tmpResVec, FSC, _, _, qVals_FDR, resolution_FDR, _ = FSCutil.FSC(self.halfMap1, self.halfMap2, maskData, self.apix, 0.143, 1, False, False, None);

		self.resolution = resolution_FDR;
		self.FSCdata = FSC;
		self.calcTTest();
		self.qVals = qVals_FDR;
		self.resVec = tmpResVec;
		self.filterMap();
		self.writeFSC();


	#---------------------------------------------
	def make_half_maps(self):

		#initialize the half maps
		tmpHalfMap1 = np.zeros((self.sizeMap, self.sizeMap));
		tmpHalfMap2 = np.zeros((self.sizeMap, self.sizeMap));
		

		#make the grid
		minX = min(np.amin(self.embeddingsHalf1[:,0]), np.amin(self.embeddingsHalf2[:,0]));
		maxX = max(np.amax(self.embeddingsHalf1[:,0]), np.amax(self.embeddingsHalf2[:,0]));
		minY = min(np.amin(self.embeddingsHalf1[:,1]), np.amin(self.embeddingsHalf2[:,1]));
		maxY = max(np.amax(self.embeddingsHalf1[:,1]), np.amax(self.embeddingsHalf2[:,1]));


		spacingX = (maxX-minX)/float(self.sizeMap -1);
		spacingY = (maxY-minY)/float(self.sizeMap -1);

		spacing = max(spacingX, spacingY);
		self.apix = spacing;

		half1 = self.embeddingsHalf1;
		half2 = self.embeddingsHalf2;
                
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

		plt.axhline(0.5, linewidth=0.5, color='r');
		plt.axhline(0.143, linewidth=0.5, color='r');
		plt.axhline(0.0, linewidth=0.5, color='b');

		plt.xlabel("1/resolution [1/A]");
		plt.ylabel("FSC");
		plt.legend();

		plt.savefig('FSC.png', dpi=300);
		plt.close();


	#---------------------------------------------
	def standardizePixelSize(self):

		#calculate convex hull
		print("Calculate smallest enclosing circle ...");
		smallestEncCirc = smallestEnclosingCircle.make_circle(np.concatenate((self.embeddingsHalf1, self.embeddingsHalf2)));

		self.encCircRad = smallestEncCirc[2];
		#scale pixel size with respect to the radius of the smallest enclosing circle
		self.apix = self.apix/self.encCircRad;


    #---------------------------------------------
	def filterMap(self):

		#fourier transform the full map
		self.filteredMap = FDRutil.lowPassFilter(np.fft.rfftn(self.fullMap), self.frequencyMap, self.apix/float(self.resolution), self.fullMap.shape);
		self.filteredMap[self.filteredMap<0.0] = 0.0;


	#---------------------------------------------
	def calcTTest(self):

		numShells = self.FSCdata.shape[0];
		sampleForTTest = self.FSCdata[int(0.8*numShells):];

		if sampleForTTest.shape[0] > 100:
			sampleForTTest = np.random.choice(sampleForTTest, 100);

		testResult = scipy.stats.ttest_1samp(sampleForTTest, 0.0);
		pVal = testResult[1];

		if pVal<0.00001:
			print("FSC is significantly deviating from 0 at high-resolutions. Points are clustered to close or sampling rate too low!")
		else:
			print("Estimated resolution at 1% FDR-FSC: {:.3f}".format(self.resolution));

