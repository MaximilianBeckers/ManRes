import numpy as np
import matplotlib.pyplot as plt
from Utilities import FSCutil,smallestEnclosingCircle

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
		self.frequencyMap = FSCutil.calculate_frequency_map(self.halfMap1);
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
	def standardizeResolution(self):

		#calculate convex hull
		print("Calculate smalles enclosing circle ...");
		hull = smallestEnclosingCircle.make_circle(self.embeddings);

		self.apix = self.apix/hull[2];


