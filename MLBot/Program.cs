using Microsoft.ML;
using Newtonsoft.Json;
using System.Diagnostics;
using System.Security.Authentication;

namespace MLBot
{
    internal class Program
    {
        static string folderPath = "data";
        static PredictionEngine<InputModel, OutputModel>? predictionEngine;

        static void Main(string[] args)
        {
            // Entrenar el modelo
            var modelBuilder = new ModelBuilder();

            if (File.Exists(Path.Combine(Directory.GetCurrentDirectory(), "model.zip")))
            {
                modelBuilder.LoadModel(Path.Combine(Directory.GetCurrentDirectory(), "model.zip"));
                predictionEngine = modelBuilder.CreatePredictionEngine();
            }
            else
            {
                modelBuilder.TrainModel(Path.Combine(Directory.GetCurrentDirectory(), folderPath, "data.tsv"));
                predictionEngine = modelBuilder.CreatePredictionEngine();
                modelBuilder.SaveModel(Path.Combine(Directory.GetCurrentDirectory(), "model.zip"));
            }

            // Interactuar con el usuario
            while (true)
            {
                Console.WriteLine("Ingrese un texto para el bot (o 'exit' para salir):");
                string userInput = Console.ReadLine();

                if (userInput.Equals("exit", StringComparison.OrdinalIgnoreCase))
                    break;

                InputModel input = new InputModel { Text = userInput };
                OutputModel prediction = predictionEngine.Predict(input);

                // Responder basándose en la predicción
                Console.WriteLine($"Predicción: {prediction.PredictedLabel}");

                var possibleResponses = GetPossibleResponses(prediction.PredictedLabel);
                // Seleccionar la respuesta con el puntaje más alto
                int bestResponseIndex = Array.IndexOf(prediction.Score, prediction.Score.Max());
                string selectedResponse = possibleResponses[bestResponseIndex];


                Console.WriteLine($"Respuestas posibles: {selectedResponse}");

            }
        }

        static string[] GetPossibleResponses(string predictedLabel)
        {
            // Simulación de la obtención de respuestas posibles desde los datos de entrenamiento
            // En un entorno real, podrías cargar esto desde una base de datos o un archivo
            return predictedLabel switch
            {
                "Greeting" => new[] { "Hi there!", "Hello!", "Hey!" },
                "Farewell" => new[] { "Goodbye!", "See you later!", "Take care!" },
                "Question" => new[] { "I'm fine, thank you!", "I'm good, how about you?", "Doing well, thanks!" },
                _ => new[] { "Sorry, I don't understand." },
            };
        }
    }
}
