using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ONNX_MNIST_Segmentation.Utils
{
    public static class Utilities
    {
        public static float[] FlattenImage(ImageSource image)
        {
            try
            {
                if (image is BitmapSource bitmap)
                {
                    int width = bitmap.PixelWidth;
                    int height = bitmap.PixelHeight;

                    byte[] flatten = new byte[height * width];

                }
            }
            catch(Exception e)
            {
                Debug.WriteLine(e);
            }
        }
    }
}
