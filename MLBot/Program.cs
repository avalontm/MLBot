using Microsoft.ML;
using Newtonsoft.Json;
using System.Security.Authentication;

namespace MLBot
{
    internal class Program
    {
        static string folderPath = "data";
        static PredictionEngine<Input, Output>? predictionEngine;

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
                modelBuilder.TrainModel(Path.Combine(Directory.GetCurrentDirectory(), folderPath));
                predictionEngine = modelBuilder.CreatePredictionEngine();
                modelBuilder.SaveModel(Path.Combine(Directory.GetCurrentDirectory(), "model.zip"));
            }

            // Conversation loop
            while (true)
            {
                Console.Write("User: ");
                var userInput = Console.ReadLine();

                if (string.IsNullOrWhiteSpace(userInput))
                {
                    break; // Exit loop if input is empty
                }

                // Predict category
                var input = new Input { Question = userInput };
                var output = predictionEngine.Predict(input);

                Console.WriteLine($"Bot: {output.PredictedLabel}");

            }
        }
    }
}
