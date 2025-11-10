using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Win32;
using ONNX_MNIST_Segmentation.ViewModels.Commands;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ONNX_MNIST_Segmentation.ViewModels
{
    public class MainVM : BaseVM
    {
        #region Image
        private ImageSource _previewImageSource;

        public ImageSource PreviewImageSource 
        {
            get => _previewImageSource;
            set
            {
                if (_previewImageSource != value)
                {
                    _previewImageSource = value;
                    NotifyPropertyChanged(nameof(PreviewImageSource));
                }
            }
        }
        #endregion

        #region Submit
        private ICommand _submitCommand;
        public ICommand SubmitCommand 
        { 
            get
            {
                if (_submitCommand == null)
                    _submitCommand = new RelayCommand(Submit);

                return _submitCommand;
            }
        }

        void Submit(object parameter)
        {
            string modelPath = "../Resources/mnist_segmentation.onnx";

            using var session = new InferenceSession(modelPath);

            var inputData = FlattenImage(PreviewImageSource);
            var inputTensor = new DenseTensor<float>(inputData, new int[] { 1, 784 });

            var inputName = session.InputMetadata.Keys.First();

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            };

            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);

            // Read output
            foreach (var result in results)
            {
                Console.WriteLine($"{result.Name}: {string.Join(", ", result.AsEnumerable<float>())}");
            }
        }
        #endregion

        #region Upload image
        private ICommand _uploadImageCommand;
        public ICommand UploadImageCommand
        {
            get
            {
                if (_uploadImageCommand == null)
                    _uploadImageCommand = new RelayCommand(UploadImage);

                return _uploadImageCommand;
            }
        }

        void UploadImage(object parameter)
        {
            var dialog = new OpenFileDialog
            {
                Title = "Select an Image",
                Filter = "Image Files|*.png;*.jpg;*.jpeg;*.bmp"
            };

            if (dialog.ShowDialog() == true)
            {
                try
                {
                    var bitmap = new BitmapImage();
                    bitmap.BeginInit();
                    bitmap.UriSource = new Uri(dialog.FileName);
                    bitmap.CacheOption = BitmapCacheOption.OnLoad;
                    bitmap.EndInit();

                    PreviewImageSource = bitmap;
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Failed to load image: {ex.Message}");
                }
            }
        }
        #endregion
    }
}
