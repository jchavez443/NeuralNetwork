using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace GenericNeuralNetwork
{
	public static class Randomizer
	{
		public static Random random = new Random();
		public static double randomGaussian(double mean, double stdDev)
		{
			//reuse this if you are generating many
			double u1 = 1.0 - random.NextDouble(); //uniform(0,1] random doubles
			double u2 = 1.0 - random.NextDouble();
			double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
			double randNormal = mean + stdDev * randStdNormal; //random normal(mean,stdDev^2)

			return randNormal;

		}

		public static double[][] shuffle(double[][] input)
		{
			double[][] output = new double[input.Length][];

			for (int i = 0; i < input.Length; i++)
			{
				output[i] = input[i];
			}

			int n = input.Length;
			double[] temp;
			while (n > 1)
			{
				n--;
				int shuffleIndex = random.Next(input.Length);

				temp = output[shuffleIndex];
				output[shuffleIndex] = output[n];
				output[n] = temp;
			}

			return output;
		}

		public static List<double[][]> shuffle(double[][] input1, double[][] input2)
		{
			double[][] output1 = new double[input1.Length][];
			double[][] output2 = new double[input2.Length][];

			for (int i = 0; i < input1.Length; i++)
			{
				output1[i] = input1[i];
			}
			for (int i = 0; i < input2.Length; i++)
			{
				output2[i] = input2[i];
			}

			int n = input1.Length;
			double[] temp;
			while (n > 1)
			{
				n--;
				int shuffleIndex = random.Next(input1.Length);

				temp = output1[shuffleIndex];
				output1[shuffleIndex] = output1[n];
				output1[n] = temp;

				temp = output2[shuffleIndex];
				output2[shuffleIndex] = output2[n];
				output2[n] = temp;
			}
			List<double[][]> ret = new List<double[][]>();
			ret.Add(output1);
			ret.Add(output2);
			return ret;
		}
	}
}
