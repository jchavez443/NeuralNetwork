using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Web;

namespace GenericNeuralNetwork
{
	public class Matrix
	{
		private double[][] matrix;
		public int x = 0;
		public int y = 0;
		public bool isVector = false;


		public Matrix()
		{

		}

		public Matrix(double[] vector)
		{

			double[][] temp = new double[1][];
			temp[0] = vector;

			matrix = temp;

			x = matrix.Length;
			y = matrix[0].Length;


			isVector = true;

		}

		public Matrix(double[][] matrix)
		{
			this.matrix = matrix;
			x = matrix.Length;
			y = matrix[0].Length;

			if (x == 1)
			{
				isVector = true;
			}
		}

		public double[][] getMatirx()
		{
			return matrix;
		}

		public double[][] multiply(Matrix left)
		{

			if (x != left.y)
				throw new Exception();

			double[][] ret = new double[left.x][];

			for (int xi = 0; xi < x; xi++)
			{
				for (int yi = 0; yi < y; yi++)
				{
					for (int i = 0; i < x; i++)
					{
						ret[xi][yi] += matrix[xi][i] * left[i][xi];
					}
				}
			}
			return ret;
		}

		public Matrix transpose()
		{
			double[][] result = new double[y][];
			for (int i = 0; i < result.Length; i++)
			{
				result[i] = new double[x];
			}



			for (int xi = 0; xi < x; xi++)
			{
				for (int yi = 0; yi < y; yi++)
				{
					result[yi][xi] = matrix[xi][yi];
				}
			}

			Matrix ret = new Matrix(result);
			return ret;
		}

		public double dotProduct(Matrix right)
		{
			if (y != 0 || right.x != 0)
			{
				throw new Exception();
			}

			if (x != right.y)
				throw new Exception();

			double scalar = 0;

			for (int i = 0; i < x; i++)
			{
				scalar += matrix[i][0] * right[0][i];
			}

			return scalar;

		}

		public Matrix hadamard(Matrix right)
		{
			if (y != right.y || x != right.x)
			{
				throw new Exception();
			}

			double[][] temp = new double[x][];

			for (int xi = 0; xi < x; xi++)
			{
				temp[xi] = new double[y];
				for (int yi = 0; yi < y; yi++)
				{
					temp[xi][yi] = matrix[xi][yi] * right[xi][yi];
				}
			}

			return new Matrix(temp);
		}

		public void showMatrix()
		{

			Debug.Write("[ ");
			for (int yi = 0; yi < y; yi++)
			{
				if (yi != 0)
					Debug.Write("| ");

				for (int xi = 0; xi < x; xi++)
				{
					Debug.Write(matrix[xi][yi] + ", ");
				}

				if (yi != y - 1)
					Debug.WriteLine("");
			}
			Debug.Write(" ]");
			Debug.WriteLine("");

		}



		//Operator Overloads
		public double[] this[int i]
		{
			get { return matrix[i]; }
			set { matrix[i] = value; }
		}

		public static Matrix operator *(Matrix left, Matrix right)
		{
			if (left.x != right.y)
				throw new Exception();

			int len = left.x;

			double[][] result = new double[right.x][];


			for (int xi = 0; xi < right.x; xi++)
			{
				result[xi] = new double[left.y];
				for (int yi = 0; yi < left.y; yi++)
				{

					for (int i = 0; i < len; i++)
					{
						result[xi][yi] += left[i][yi] * right[xi][i];
					}
				}
			}

			Matrix ret = new Matrix(result);
			return ret;
		}

		public static Matrix operator *(double left, Matrix right)
		{

			double[][] result = new double[right.x][];


			for (int xi = 0; xi < right.x; xi++)
			{
				result[xi] = new double[right.y];
				for (int yi = 0; yi < right.y; yi++)
				{
					result[xi][yi] = left * right[xi][yi];
				}
			}

			Matrix ret = new Matrix(result);
			return ret;
		}

		public static Matrix operator *(Matrix right, double left)
		{

			double[][] result = new double[right.x][];


			for (int xi = 0; xi < right.x; xi++)
			{
				result[xi] = new double[right.y];
				for (int yi = 0; yi < right.y; yi++)
				{
					result[xi][yi] = left * right[xi][yi];
				}
			}

			Matrix ret = new Matrix(result);
			return ret;
		}

		public static Matrix operator +(Matrix left, Matrix right)
		{
			if (left.x != right.x || left.y != right.y)
				throw new Exception();

			double[][] result = new double[left.x][];

			for (int xi = 0; xi < left.x; xi++)
			{
				result[xi] = new double[left.y];
				for (int yi = 0; yi < left.y; yi++)
				{
					result[xi][yi] = right[xi][yi] + left[xi][yi];
				}
			}

			Matrix ret = new Matrix(result);
			return ret;
		}

		public static double[] operator +(double[] left, Matrix right)
		{
			if (right.x != 1 || left.Length != right.y)
				throw new Exception();

			double[] result = new double[left.Length];

			for (int xi = 0; xi < left.Length; xi++)
			{
				result[xi] = left[xi] + right[0][xi];
			}

			//Matrix ret = new Matrix(result);
			return result;
		}

		public static double[] operator +(Matrix right, double[] left)
		{
			if (right.x != 1 || left.Length != right.y)
				throw new Exception();

			double[] result = new double[left.Length];

			for (int xi = 0; xi < left.Length; xi++)
			{
				result[xi] = left[xi] + right[0][xi];
			}

			//Matrix ret = new Matrix(result);
			return result;
		}

		public static double[][] operator +(double[][] left, Matrix right)
		{
			if (right.x != left.Length)
				throw new Exception();

			double[][] result = new double[left.Length][];

			for (int xi = 0; xi < left.Length; xi++)
			{
				if (left.Length != right.y)
					throw new Exception();

				result[xi] = new double[left[xi].Length];
				for (int yi = 0; yi < right.y; yi++)
				{
					result[xi][yi] = left[xi][yi] + right[xi][xi];
				}
			}

			//Matrix ret = new Matrix(result);
			return result;
		}

		public static double[][] operator +(Matrix right, double[][] left)
		{
			if (right.x != left.Length)
				throw new Exception();

			double[][] result = new double[left.Length][];

			for (int xi = 0; xi < left.Length; xi++)
			{
				if (left[xi].Length != right.y)
					throw new Exception();

				result[xi] = new double[left[xi].Length];
				for (int yi = 0; yi < right.y; yi++)
				{
					result[xi][yi] = left[xi][yi] + right[xi][xi];
				}
			}

			//Matrix ret = new Matrix(result);
			return result;
		}

		public static Matrix operator -(Matrix left, Matrix right)
		{
			if (left.x != right.x || left.y != right.y)
				throw new Exception();

			double[][] result = new double[left.x][];

			for (int xi = 0; xi < left.x; xi++)
			{
				result[xi] = new double[left.y];
				for (int yi = 0; yi < left.y; yi++)
				{
					result[xi][yi] = right[xi][yi] - left[xi][yi];
				}
			}

			Matrix ret = new Matrix(result);
			return ret;
		}

		public static double[] operator -(double[] left, Matrix right)
		{
			if (right.x != 1 || left.Length != right.y)
				throw new Exception();

			double[] result = new double[left.Length];

			for (int xi = 0; xi < left.Length; xi++)
			{
				result[xi] = left[xi] - right[0][xi];
			}

			//Matrix ret = new Matrix(result);
			return result;
		}

		public static double[] operator -(Matrix left, double[] right)
		{
			if (left.x != 1 || left.y != right.Length)
				throw new Exception();

			double[] result = new double[right.Length];

			for (int xi = 0; xi < right.Length; xi++)
			{
				result[xi] = left[0][xi] - right[xi];
			}

			//Matrix ret = new Matrix(result);
			return result;
		}

		public static double[][] operator -(double[][] left, Matrix right)
		{
			if (right.x != left.Length)
				throw new Exception();

			double[][] result = new double[left.Length][];

			for (int xi = 0; xi < left.Length; xi++)
			{
				if (left[xi].Length != right.y)
					throw new Exception();

				result[xi] = new double[left[xi].Length];
				for (int yi = 0; yi < right.y; yi++)
				{
					result[xi][yi] = left[xi][yi] - right[xi][xi];
				}
			}

			//Matrix ret = new Matrix(result);
			return result;
		}

		public static double[][] operator -(Matrix left, double[][] right)
		{
			if (left.x == right.Length)
				throw new Exception();

			double[][] result = new double[right.Length][];

			for (int xi = 0; xi < right.Length; xi++)
			{

				if (right[xi].Length != left.y)
					throw new Exception();

				result[xi] = new double[left[xi].Length];
				for (int yi = 0; yi < left.y; yi++)
				{
					result[xi][yi] = left[xi][yi] - right[xi][xi];
				}
			}

			//Matrix ret = new Matrix(result);
			return result;
		}


	}
}