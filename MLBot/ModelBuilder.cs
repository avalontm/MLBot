using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.VisualBasic;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLBot
{
    public class UserConversation
    {
        public string? UserInput { get; set; }
        public string? BotResponse { get; set; }
    }

    public class Input
    {
        public string Category { get; set; }
        public string Question { get; set; }
        public string Answer { get; set; }
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
        DataViewSchema? modelSchema;

        public ModelBuilder()
        {
            _mlContext = new MLContext();
        }

        public void TrainModel(string folderPath)
        {
            // Load training data from YAML files
            var trainingData = ConversationLoader.LoadConversationsFromFolder(folderPath);

            // Convert input data to IEnumerable<Input>
            var inputData = trainingData
                .SelectMany(group => group.Answers.Select(answer => new Input
                {
                    Category = group.Category,
                    Question = group.Question,
                    Answer = answer
                }))
                .ToList();

            // Load input data into an IDataView
            _dataView = _mlContext.Data.LoadFromEnumerable(inputData);

            // Define the transformation and training pipeline
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey(nameof(Input.Answer))
                .Append(_mlContext.Transforms.Text.FeaturizeText("Features", nameof(Input.Question)))
                .Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: nameof(Input.Answer)))
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
            _model = _mlContext.Model.Load(modelPath, out modelSchema);
        }

        public PredictionEngine<Input, Output> CreatePredictionEngine()
        {
            return _mlContext.Model.CreatePredictionEngine<Input, Output>(_model);
        }
    }
}
