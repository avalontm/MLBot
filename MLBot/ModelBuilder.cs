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
    public class InputModel
    {
        [LoadColumn(0)]
        public string Label { get; set; }

        [LoadColumn(1)]
        public string Text { get; set; }

        [LoadColumn(2)]
        public string Responses { get; set; }
    }

    public class OutputModel
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabel { get; set; }

        [ColumnName("PredictedResponses")]
        public string PredictedResponses { get; set; }
        public float[] Score { get; set; }
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

        public void TrainModel(string filePath)
        {
            // Cargar datos
            _dataView = _mlContext.Data.LoadFromTextFile<InputModel>(filePath, separatorChar: '\t', hasHeader: true, allowSparse: false);

            // Preparar el pipeline de transformación y entrenamiento
            var pipeline = _mlContext.Transforms.Conversion.MapValueToKey("Label", "Label")
               .Append(_mlContext.Transforms.Text.FeaturizeText("TitleFeaturized", "Text"))
               .Append(_mlContext.Transforms.Text.FeaturizeText("ResponsesFeaturized", "Responses"))
               .Append(_mlContext.Transforms.Concatenate("Features", "TitleFeaturized", "ResponsesFeaturized"))
               .AppendCacheCheckpoint(_mlContext);

            var trainer = _mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features");

            var trainingPipeline = pipeline
                .Append(trainer)
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Entrenar el modelo
            Console.WriteLine("Training model...");
            _model = trainingPipeline.Fit(_dataView);
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

        public PredictionEngine<InputModel, OutputModel> CreatePredictionEngine()
        {
            return _mlContext.Model.CreatePredictionEngine<InputModel, OutputModel>(_model);
        }
    }
}
