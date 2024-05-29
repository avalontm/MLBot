using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
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
        IDataView? dataView;

        public ModelBuilder()
        {
            _mlContext = new MLContext();
        }

        public void TrainModel(string folderPath)
        {
            // Load training data from YAML files in the folder
            IEnumerable<Conversation> trainingData = ConversationLoader.LoadConversationsFromFolder(folderPath);

            // Verify that we have at least 2 distinct labels
            var distinctCategories = trainingData.SelectMany(c => c.categories).Distinct().ToList();
            if (distinctCategories.Count < 2)
            {
                throw new InvalidOperationException("Training data must contain at least two distinct categories.");
            }

            // Convert input data to IEnumerable<Input>
            var inputData = trainingData
                .SelectMany(c => c.conversations.SelectMany(conv => conv.Select(conversation => new Input
                {
                    Text = conversation,
                    Label = c.categories.FirstOrDefault() // Assuming each conversation belongs to a single category
                })));

            // Load input data into an IDataView
            dataView = _mlContext.Data.LoadFromEnumerable(inputData);

            // Define the transformation and training pipeline
            var pipeline = _mlContext.Transforms.Text.FeaturizeText("Features", nameof(Input.Text))
                .Append(_mlContext.Transforms.Conversion.MapValueToKey(nameof(Input.Label)))
                .Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy())
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Train the model
            Console.WriteLine("Training model...");
            _model = pipeline.Fit(dataView);
        }

        public void SaveModel(string modelPath)
        {
            _mlContext.Model.Save(_model, dataView.Schema, modelPath);
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
