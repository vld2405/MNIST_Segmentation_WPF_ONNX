using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ONNXSegmentationWPF.Models
{
    public class OnnxModel : IDisposable
    {
        private readonly InferenceSession _session;

        public OnnxModel(string modelPath)
        {
            if (!File.Exists(modelPath))
                throw new FileNotFoundException("ONNX model not found", modelPath);

            _session = new InferenceSession(modelPath);
        }

        public BitmapImage RunInference(BitmapImage inputImage)
        {
            int width = 28;
            int height = 28;

            var wbSource = new WriteableBitmap(inputImage);
            var converted = new FormatConvertedBitmap(wbSource, PixelFormats.Gray8, null, 0);
            var wb = new WriteableBitmap(converted);

            int srcWidth = wb.PixelWidth;
            int srcHeight = wb.PixelHeight;
            int bytesPerPixel = wb.Format.BitsPerPixel / 8;
            int stride = srcWidth * bytesPerPixel;
            byte[] pixelData = new byte[stride * srcHeight];
            wb.CopyPixels(pixelData, stride, 0);

            var tensor = new DenseTensor<float>(new[] { 1, 1, height, width });

            double scaleX = (double)srcWidth / width;
            double scaleY = (double)srcHeight / height;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int srcX = (int)(x * scaleX);
                    int srcY = (int)(y * scaleY);
                    int srcIndex = (srcY * srcWidth + srcX) * bytesPerPixel;

                    if (srcIndex >= pixelData.Length)
                        continue;

                    byte gray = pixelData[srcIndex];
                    tensor[0, 0, y, x] = gray / 255f;
                }
            }

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", tensor)
            };

            using var results = _session.Run(inputs);
            var output = results.First().AsTensor<float>();

            var outBmp = new WriteableBitmap(width, height, 96, 96, PixelFormats.Gray8, null);
            byte[] outPixels = new byte[width * height];

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    float val = output[0, 0, y, x];
                    byte gray = (byte)(Math.Clamp(val, 0f, 1f) * 255);
                    outPixels[y * width + x] = gray;
                }
            }

            outBmp.WritePixels(new System.Windows.Int32Rect(0, 0, width, height), outPixels, width, 0);

            var encoder = new PngBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create(outBmp));
            using var ms = new MemoryStream();
            encoder.Save(ms);
            ms.Position = 0;

            var resultImage = new BitmapImage();
            resultImage.BeginInit();
            resultImage.CacheOption = BitmapCacheOption.OnLoad;
            resultImage.StreamSource = ms;
            resultImage.EndInit();
            resultImage.Freeze();

            return resultImage;
        }

        public void Dispose() => _session?.Dispose();
    }
}
