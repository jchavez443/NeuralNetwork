using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Threading;


namespace GenericNeuralNetwork
{
	public class Network
	{

		public static Random randGlobal = new Random();

		public int L;
		public int[] sizes;
		public List<Matrix> weightsMat;
		public List<Matrix> baisisMat;

		public Network(int[] sizes)
		{
			L = sizes.Length;
			this.sizes = sizes;
			weightsMat = assignWeights(sizes);
			baisisMat = assignBaisis(sizes);
			
		}

		private List<Matrix> assignWeights(int[] sizes)
		{
			List<Matrix> weights = new List<Matrix>();
			
			double ran = 0;

			for (int l = 1; l < sizes.Length; l++)
			{
				double[][] w = new double[sizes[l]][];
				for (int x = 0; x < sizes[l]; x++)
				{
					w[x] = new double[sizes[l - 1]];
					for (int y = 0; y < sizes[l - 1]; y++)
					{
						ran = Randomizer.randomGaussian(0, 1);
						w[x][y] = ran;
					}
				}
				weights.Add(new Matrix(w));
			}

			return weights;
		}

		private List<Matrix> assignBaisis(int[] sizes)
		{
			List < Matrix > baisis = new List<Matrix>();
			
			double ran = 0;

			for (int l = 1; l < sizes.Length; l++)
			{
				double[] b = new double[sizes[l]];
				for (int x = 0; x < b.Length; x++)
				{
					ran = Randomizer.randomGaussian(0, 1);
					b[x] = ran;
				}
				baisis.Add(new Matrix(b));
			}
			return baisis;
		}




		public double[] feedForward(double[] a)
		{

			Matrix current = new Matrix(a);
			for (int l = 0; l < sizes.Length - 1; l++)
			{
				current = weightsMat[l].transpose() * current + baisisMat[l];
				current = sigmoid(current);
			}
			return current.getMatirx()[0];
		}



		public void SGD(double[][] trainingData, double[][] trainingDataExpected, double epochs, int miniBatchSize, double learningRate, bool threading)
		{
			int n = trainingData.Length;
			int epochCount = 0;

			double[][] samples = new double[miniBatchSize][];
			double[][] samplesExpected = new double[miniBatchSize][];

			double[][] epochSamplesShuffle = new double[trainingData.Length][];
			double[][] epochSamplesExpectedShuffle = new double[trainingDataExpected.Length][];

			List<double[][]> shuffledList = new List<double[][]>();

			
			for (int i = 0; i < epochs; i++)
			{
				epochCount = 0;
				shuffledList = Randomizer.shuffle(trainingData, trainingDataExpected);
				epochSamplesShuffle = shuffledList[0];
				epochSamplesExpectedShuffle = shuffledList[1];

				while (epochCount < n)
				{

					int count = 0;
					while (count < miniBatchSize)
					{
						samples[count] = epochSamplesShuffle[epochCount];
						samplesExpected[count] = epochSamplesExpectedShuffle[epochCount];
						count++;
						epochCount++;
					}
					updateMiniBatch(samples, samplesExpected, learningRate, miniBatchSize);
					
				}	
				evaluate(i);
			}

		}


		public static void test(object a, object b)
		{

		}

		public List<Matrix>[] backprop(double[] a, double[] expected)
		{

			Matrix expectedm = new Matrix(expected);
			Matrix activation = new Matrix(a);

			List<Matrix>  deltaBM = new List<Matrix>();
			List<Matrix>  deltaWM = new List<Matrix>();

			List<Matrix> activations = new List<Matrix>();
			activations.Add(activation);
			List<Matrix> zs = new List<Matrix>();
			Matrix z = null;
			List<Matrix> errors = new List<Matrix>();

			//Forward pass
			for (int l = 0; l < sizes.Length - 1; l++)
			{
				z = weightsMat[l].transpose() * activation + baisisMat[l];
				zs.Add(z);
				activation = sigmoid(z);
				activations.Add(activation);

				if (l != 0)
					errors.Add(new Matrix());
			}

			//Backwards Pass
			Matrix errorLm = costDerivative(activations[activations.Count - 1], expectedm).hadamard(sigmoidPrime(z));
			errors.Add(errorLm);

			deltaBM.Add(errorLm);
			deltaWM.Add(activations[activations.Count - 2] * errorLm.transpose());

			for (int l = sizes.Length - 2; l > 0; l--)
			{
				z = zs[l - 1];
				Matrix errorm = (weightsMat[l] * errors[l]).hadamard(sigmoidPrime(z));
				errors[l - 1] = errorm;
				deltaBM.Insert(0, errorm);
				deltaWM.Insert(0, activations[activations.Count - 2 - l] * errorm.transpose());
			}

			List<Matrix>[] ret = new List<Matrix>[2];
			ret[0] = deltaBM;
			ret[1] = deltaWM;
			return ret;
		}

		public void updateMiniBatch(double[][] miniBatch, double[][] expected, double eta, int miniBatchLen)
		{

			List<Matrix> totalDeltaBM = new List<Matrix>();
			List<Matrix> totalDeltaWM = new List<Matrix>();
			for (int i = 0; i < baisisMat.Count; i++)
			{
				totalDeltaBM.Add(new Matrix(new double[baisisMat[i].y]));
			}

			for (int i = 0; i < weightsMat.Count; i++)
			{
				double[][] init = new double[weightsMat[i].x][];
				for (int x = 0; x < weightsMat[i].x; x++)
				{
					init[x] = new double[weightsMat[i].y];
				}
				Matrix temp = new Matrix(init);
				totalDeltaWM.Add(temp);
			}


			for (int i = 0; i < miniBatch.Length; i++)
			{
				List<Matrix>[] deltas = backprop(miniBatch[i], expected[i]);
				List<Matrix> deltaBM = deltas[0];
				List<Matrix> deltaWM = deltas[1];
				for (int b = 0; b < baisisMat.Count; b++)
				{
					totalDeltaBM[b] += deltaBM[b];
				}

				for (int w = 0; w < weightsMat.Count; w++)
				{
					totalDeltaWM[w] += deltaWM[w];
				}
			}

			for (int b = 0; b < baisisMat.Count; b++)
			{
				baisisMat[b] = baisisMat[b] - ((eta / miniBatchLen) * totalDeltaBM[b]);
			}

			for (int w = 0; w < weightsMat.Count; w++)
			{
				weightsMat[w] = weightsMat[w] - ((eta / miniBatchLen) * totalDeltaWM[w]);
			}
		}


		public Matrix costDerivative(Matrix outPutActivations, Matrix expected)
		{
			//This is only for parabolic cost funtion (a - y)^2
			double[] init = new double[outPutActivations.y];
			Matrix ret = new Matrix(init);
			for (var i = 0; i < outPutActivations.y; i++)
			{
				ret[0][i] = outPutActivations[0][i] - expected[0][i];
			}
			return ret;
		}

		public Matrix sigmoid(Matrix z)
		{

			if (!z.isVector)
				throw new Exception();

			double[] a = new double[z.y];

			for (int i = 0; i < z.y; i++)
			{
				a[i] = 1 / (1 + Math.Exp(-z[0][i]));
			}

			Matrix temp = new Matrix(a);
			return temp;
		}

		public Matrix sigmoidPrime(Matrix z)
		{
			if (!z.isVector)
				throw new Exception();

			double[] a = new double[z.y];
			//a = sigmoid(z) * (1 - sigmoid(z));


			Matrix left = sigmoid(z);


			for (int i = 0; i < left.y; i++)
			{
				a[i] = left[0][i] * (1 - left[0][i]);
			}


			return new Matrix(a);
		}

		public void evaluate(int epoch)
		{
			double[] output;
			int number;
			double[] real;
			int actual;
			double count = 0;
			double percent = 0; ;
			for (int i = 0; i < 10000; i++)
			{

				int index = randGlobal.Next(60000 - 1);
				output = feedForward(Program.imagesD[index]);
				number = Program.findMaxIndex(output);

				real = Program.labelsD[index];
				actual = Program.findMaxIndex(real);

				if (actual == number)
				{
					count++;
				}
				percent = count / (i + 1);
			}
			percent = percent * 100;
			Console.WriteLine("Completed " + epoch + "     " + percent + "%");
		}


	}
}