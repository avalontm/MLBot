using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLBot
{
    public class Input
    {
        public string? Text { get; set; }
        public string? Label { get; set; }
    }

    public class Output
    {
        public string? PredictedLabel { get; set; }
        public float[]? Score { get; set; }
        public float Probability { get; set; }
    }

    public class ModelBuilder
    {
        private readonly MLContext _mlContext;
        private ITransformer? _model;
        IDataView? _dataView;

        public ModelBuilder()
        {
            _mlContext = new MLContext();
        }

        public void TrainModel(string filePath)
        {
            // Load training data from YAML file
            IEnumerable<Conversation> trainingData = ConversationLoader.LoadConversationsFromFolder(filePath);

            // Ensure we have at least 2 distinct labels
            var distinctCategories = trainingData.Select(c => c.Category).Distinct().ToList();
            if (distinctCategories.Count < 2)
            {
                throw new InvalidOperationException("Training data must contain at least two distinct categories.");
            }

            // Convert input data to IEnumerable<Input>
            var inputData = trainingData.Select(c => new Input { Text = c.Text, Label = c.Category });

            // Load input data into an IDataView
            _dataView = _mlContext.Data.LoadFromEnumerable(inputData);

            // Define the transformation and training pipeline
            var pipeline = _mlContext.Transforms.Text.FeaturizeText("Features", nameof(Input.Text))
                .Append(_mlContext.Transforms.Conversion.MapValueToKey(nameof(Input.Label)))
                .Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy())
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Train the model
            Console.WriteLine("Training model...");
            _model = pipeline.Fit(_dataView);
        }

        public void SaveModel(string modelPath)
        {
            _mlContext.Model.Save(_model, _dataView.Schema, modelPath);
            Console.WriteLine($"Model saved to: {modelPath}");
        }

        public void LoadModel(string modelPath)
        {
            DataViewSchema modelSchema;
            _model = _mlContext.Model.Load(modelPath, out modelSchema);
        }

        public PredictionEngine<Input, Output> CreatePredictionEngine()
        {
            return _mlContext.Model.CreatePredictionEngine<Input, Output>(_model);
        }
    }
}
