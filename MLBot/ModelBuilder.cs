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
            // Cargar los datos de entrenamiento desde los archivos YAML en la carpeta
            IEnumerable<Conversation> trainingData = ConversationLoader.LoadConversationsFromFolder(folderPath);

            // Convertir las categorías a una cadena única separada por comas
            var categoriesAsString = string.Join(",", trainingData.SelectMany(c => c.categories).Distinct());

            // Convertir los datos de entrada a un IEnumerable<Input>
            var inputData = trainingData
                .SelectMany(c => c.conversations.SelectMany(conv => conv)) // Combina todas las conversaciones en una sola secuencia
                .Select(conversation => new Input { Text = conversation, Label = categoriesAsString });

            // Cargar los datos de entrada en un IDataView
            dataView = _mlContext.Data.LoadFromEnumerable(inputData);

            // Definir el pipeline de transformación y entrenamiento
            var pipeline = _mlContext.Transforms.Text.FeaturizeText("Features", nameof(Input.Text))
                .Append(_mlContext.Transforms.Conversion.MapValueToKey(nameof(Input.Label)))
                .Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy())
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Crear un objeto Progress para monitorear el progreso del entrenamiento
            var progressHandler = new Progress<MulticlassClassificationMetrics>(p =>
            {
                // Puedes imprimir información de progreso aquí, por ejemplo:
                Console.WriteLine($"Iteración: {p.MicroAccuracy}, Pérdida: {p.LogLoss}");
            });

            // Entrenar el modelo
            Console.WriteLine("Entrenando modelo...");
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
