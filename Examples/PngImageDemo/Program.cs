using ConvNetSharp;
using ConvNetSharp.Layers;
using ConvNetSharp.Training;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Text;
using System.Threading.Tasks;
using System.Xml;

namespace PngImageDemo
{
    class Program
    {
        private const int BatchSize = 30;
        private readonly Random random = new Random();
        private readonly CircularBuffer<double> trainAccWindow = new CircularBuffer<double>(100);
        private readonly CircularBuffer<double> valAccWindow = new CircularBuffer<double>(100);
        private readonly CircularBuffer<double> wLossWindow = new CircularBuffer<double>(100);
        private readonly CircularBuffer<double> xLossWindow = new CircularBuffer<double>(100);
        private Net net;
        private int stepCount;
        private List<PngImageEntry> testing;
        private AdadeltaTrainer trainer;
        private List<PngImageEntry> training;
        private int trainingCount = BatchSize;

        private int imageWidth = 13;
        private int imageHeight = 89;

        static void Main(string[] args)
        {
            Program program = new PngImageDemo.Program();
            program.RunDemo();
        }

        private void RunDemo()
        {
            this.training = PngImageReader.Load("Image_Finger", "Image_NoDefect", @"C:\ClassifierLocalDefectDetectionStrategy");
            this.testing = PngImageReader.Load("Image_Finger", "Image_NoDefect", @"C:\ClassifierLocalDefectDetectionStrategy");
            Console.WriteLine($"Test data loaded. {this.training?.Count} items");
            // Create network
            this.net = new Net();
            this.net.AddLayer(new InputLayer(imageWidth - 4, imageHeight - 4, 1));
            this.net.AddLayer(new ConvLayer(5, 5, 8) { Stride = 1, Pad = 2});
            this.net.AddLayer(new PoolLayer(2, 2) { Stride = 2 });
            this.net.AddLayer(new ConvLayer(5, 5, 16) { Stride = 1, Pad = 2});
            this.net.AddLayer(new PoolLayer(3, 3) { Stride = 3 });
            this.net.AddLayer(new FullyConnLayer(2));
            this.net.AddLayer(new SoftmaxLayer(2));

            DataContractSerializer serializer = new DataContractSerializer(typeof(Net));
            using (XmlReader reader = XmlReader.Create(@"C:\Projects\test.xml"))
            {
           //     this.net = (Net)serializer.ReadObject(reader);
            }

            this.trainer = new AdadeltaTrainer(this.net)
            {
                BatchSize = 20,
                L2Decay = 0.001,
            };

            foreach (var sample in this.GetAllTestSamples())
            {
                var x = sample.Volume;
                double y = sample.IsDefect ? 1.0 : 0.0;

                // use x to build our estimate of validation error
                this.net.Forward(x);
                var yhat = this.net.GetPrediction();
                var valAcc = yhat == y ? 1.0 : 0.0;
                this.valAccWindow.Add(valAcc);
            }

            Console.WriteLine("Convolutional neural network learning...[Press any key to stop]");
            do
            {
                var sample = this.SampleTrainingInstance();
                this.Step(sample);
            } while (!Console.KeyAvailable);

            var settings = new XmlWriterSettings { Indent = true };
            using (XmlWriter myWriter = XmlWriter.Create(@"c:\Projects\test.xml", settings))
            {
                serializer.WriteObject(myWriter, this.net);
            }
        }

        private void Step(Item sample)
        {
            var x = sample.Volume;
            double y = sample.IsDefect ? 1.0 : 0.0;

            if (sample.IsValidation)
            {
                // use x to build our estimate of validation error
                this.net.Forward(x);
                var yhat = this.net.GetPrediction();
                var valAcc = yhat == y ? 1.0 : 0.0;
                this.valAccWindow.Add(valAcc);
                return;
            }

            // train on it with network
            this.trainer.Train(x, y);
            var lossx = this.trainer.CostLoss;
            var lossw = this.trainer.L2DecayLoss;

            // keep track of stats such as the average training error and loss
            var prediction = this.net.GetPrediction();
            var trainAcc = prediction == y ? 1.0 : 0.0;
            this.xLossWindow.Add(lossx);
            this.wLossWindow.Add(lossw);
            this.trainAccWindow.Add(trainAcc);

            if (this.stepCount % 200 == 0)
            {
                if (this.xLossWindow.Count == this.xLossWindow.Capacity)
                {
                    var xa = this.xLossWindow.Items.Average();
                    var xw = this.wLossWindow.Items.Average();
                    var loss = xa + xw;

                    Console.WriteLine("Loss: {0} Train accuracy: {1}% Test accuracy: {2}%", loss,
                        Math.Round(this.trainAccWindow.Items.Average() * 100.0, 2),
                        Math.Round(this.valAccWindow.Items.Average() * 100.0, 2));

                    Console.WriteLine("Example seen: {0} Fwd: {1}ms Bckw: {2}ms", this.stepCount,
                        Math.Round(this.trainer.ForwardTime.TotalMilliseconds, 2),
                        Math.Round(this.trainer.BackwardTime.TotalMilliseconds, 2));
                }
            }

            if (this.stepCount % 1000 == 0)
            {
                this.TestPredict();
            }

            this.stepCount++;
        }

        private void TestPredict()
        {
            for (var i = 0; i < 50; i++)
            {
                List<Item> sample = this.SampleTestingInstance();
                var y = sample[0].IsDefect; // ground truth label

                // forward prop it through the network
                var average = new Volume(1, 1, 2, 0.0);
                var n = sample.Count;
                for (var j = 0; j < n; j++)
                {
                    var a = this.net.Forward(sample[j].Volume);
                    average.AddFrom(a);
                }

                var predictions = average.Select((w, k) => new { Label = k, Weight = w }).OrderBy(o => -o.Weight);
            }
        }

        private Item SampleTrainingInstance()
        {
            var n = this.random.Next(this.trainingCount);
            var entry = training[n];

            // load more batches over time
            if (stepCount % 5 == 0 && stepCount > 0)
            {
                trainingCount = Math.Min(trainingCount + BatchSize, training.Count);
            }

            // Create volume from image data
            var x = new Volume(imageWidth, imageHeight, 1, 0.0);

            for (var i = 0; i < imageHeight; i++)
            {
                for (var j = 0; j < imageWidth; j++)
                {
                    x.Set(j + i * imageWidth, entry.Image[j + i * imageWidth] / 255.0);
                }
            }

            //x = x.Augment(2);

            return new Item { Volume = x, IsDefect = entry.IsDefect, IsValidation = n % 10 == 0 };
        }

        private List<Item> SampleTestingInstance()
        {
            var result = new List<Item>();
            var n = this.random.Next(this.testing.Count);
            var entry = this.testing[n];

            // Create volume from image data
            var x = new Volume(imageWidth, imageHeight, 1, 0.0);

            for (var i = 0; i < imageHeight; i++)
            {
                for (var j = 0; j < imageWidth; j++)
                {
                    x.Set(j + i * imageWidth, entry.Image[j + i * imageWidth] / 255.0);
                }
            }

            //            for (var i = 0; i < 4; i++)
            {
                result.Add(new Item { Volume = x/*.Augment(2)*/, IsDefect = entry.IsDefect });
            }

            return result;
        }

        private List<Item> GetAllTestSamples()
        {
            var result = new List<Item>();
            foreach (var entry in this.testing)
            {
                // Create volume from image data
                var x = new Volume(imageWidth, imageHeight, 1, 0.0);

                for (var i = 0; i < imageHeight; i++)
                {
                    for (var j = 0; j < imageWidth; j++)
                    {
                        x.Set(j + i * imageWidth, entry.Image[j + i * imageWidth] / 255.0);
                    }
                }

                //                for (var i = 0; i < 4; i++)
                {
                    result.Add(new Item { Volume = x/*.Augment(2)*/, IsDefect = entry.IsDefect });
                }
            }

            return result;
        }

        private class Item
        {
            public Volume Volume { get; set; }

            public bool IsDefect { get; set; }

            public bool IsValidation { get; set; }
        }

    }
}
