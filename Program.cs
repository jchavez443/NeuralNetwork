using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Threading;


namespace GenericNeuralNetwork
{
	class Program
	{

		public static double[][] imagesD;
		public static double[][] labelsD;
		public static int threadCount = 0;
		public static Mutex mutex = new Mutex();
		static void Main(string[] args)
		{


			string currentSolutionDir = Directory.GetParent(Directory.GetCurrentDirectory()).Parent.Parent.FullName;
			string directoryPath = currentSolutionDir + "\\Training Images";
			DirectoryInfo directorySelected = new DirectoryInfo(directoryPath);
			FileInfo[] files = directorySelected.GetFiles("*.gz");



			/*
			 * About the files,
			 * The Files are Most significat byte first edian, 
			 * the images are grey scalled 28 * 28 = 784 pixels.
			 * there are 60,000 images.
			 * the other file is filled with labels 0 - 9
			 * the indexs match
			 * */

			byte[] labels = null;
			byte[] images = null;

			foreach (FileInfo fileToDecompress in files)
			{
				using (FileStream origFileStream = fileToDecompress.OpenRead())
				{
					using (FileStream fileStream = File.Create(directoryPath + "\\testDcompression.txt"))
					{
						using (GZipStream GZ = new GZipStream(origFileStream, CompressionMode.Decompress))
						{
							if (fileToDecompress.FullName.Contains("images"))
							{
								int index = 47040000 + 16; //16 is the file offset the other number is 28 * 28 * 60,000
								images = new byte[index];
								GZ.Read(images, 0, index);
							}
							else
							{
								int index = 60008;
								labels = new byte[index];
								GZ.Read(labels, 0, index);
							}
						}
					}
				}
			}

			imagesD = new double[60000][];
			labelsD = new double[60000][];


			for (int i = 0; i < imagesD.Length; i++)
			{
				int start = (16 + i * 784); // offset and image sizes
				imagesD[i] = new double[784];
				Array.Copy(images, start, imagesD[i], 0, 784);
				for (int y = 0; y < imagesD[i].Length; y++)
				{
					imagesD[i][y] = imagesD[i][y] / 255; //MUST scale the inputs 0 - 1
				}
			}

			for (int i = 0; i < labelsD.Length; i++)
			{
				labelsD[i] = new double[10];
				labelsD[i][labels[8 + i]] = 1;
			}

			labels = null;
			images = null;


			//initialize the network
			int[] sizes = { 784, 30, 10 };
			Network net = new Network(sizes);
			
			//Train the Network
			net.SGD(imagesD, labelsD, 30, 10, 3, true);




			Random ran = new Random();
			double percent = 0;
			double count = 0;
			double[] output;
			double[] real;
			int actual;
			int number;
			for (int i = 0; i < 40; i++)
			{

				int index = ran.Next(60000 - 1);
				output = net.feedForward(imagesD[index]);
				number = findMaxIndex(output);

				real = labelsD[index];
				actual = findMaxIndex(real);

				if (actual == number)
				{
					count++;
				}
				percent = count / (i + 1);
			}


		}
		public static void displayImage(double[] image, int x, int y)
		{
			int i = 0;
			for (int yi = 0; yi < y; yi++)
			{
				for (int xi = 0; xi < x; xi++)
				{

					if (image[i] > 0)
					{
						Console.Write(".");
					}
					else
					{
						Console.Write(0);
					}

					i++;
				}
				Console.WriteLine("");
			}

		}

		public static int findMaxIndex(double[] input)
		{
			int number = 0;
			double max = -1;
			for (int x = 0; x < input.Length; x++)
			{
				if (input[x] > max)
				{
					number = x;
					max = input[x];
				}

			}
			return number;
		}
	}
}
