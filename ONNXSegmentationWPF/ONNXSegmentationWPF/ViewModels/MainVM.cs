using Microsoft.Win32;
using System;
using System.ComponentModel;
using System.IO;
using System.Windows.Input;
using System.Windows.Media.Imaging;
using ONNXSegmentationWPF.Models;
using ONNXSegmentationWPF.ViewModels.Commands;

namespace ONNXSegmentationWPF.ViewModels
{
    public class MainVM : INotifyPropertyChanged
    {
        private BitmapImage _inputImage;
        private BitmapImage _outputImage;
        private readonly OnnxModel _model;

        public BitmapImage InputImage
        {
            get => _inputImage;
            set { _inputImage = value; OnPropertyChanged(nameof(InputImage)); }
        }

        public BitmapImage OutputImage
        {
            get => _outputImage;
            set { _outputImage = value; OnPropertyChanged(nameof(OutputImage)); }
        }

        public ICommand UploadCommand { get; }
        public ICommand SubmitCommand { get; }

        public MainVM()
        {
            string modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "mnist_segmentation.onnx");
            _model = new OnnxModel(modelPath);

            UploadCommand = new RelayCommand(_ => UploadImage());
            SubmitCommand = new RelayCommand(_ => RunInference(), _ => InputImage != null);
        }

        private void UploadImage()
        {
            var dlg = new OpenFileDialog { Filter = "Image Files|*.jpg;*.png;*.bmp" };
            if (dlg.ShowDialog() == true)
            {
                var img = new BitmapImage();
                img.BeginInit();
                img.CacheOption = BitmapCacheOption.OnLoad;
                img.UriSource = new Uri(dlg.FileName);
                img.EndInit();
                img.Freeze();
                InputImage = img;
            }
        }

        private void RunInference()
        {
            if (InputImage == null) return;
            OutputImage = _model.RunInference(InputImage);
        }

        public event PropertyChangedEventHandler PropertyChanged;
        private void OnPropertyChanged(string name)
            => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
    }
}
