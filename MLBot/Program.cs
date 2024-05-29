using System.Security.Authentication;

namespace MLBot
{
    internal class Program
    {
        static string folderPath = "data";

        static void Main(string[] args)
        {
            // Entrenar el modelo
            var modelBuilder = new ModelBuilder();
            modelBuilder.TrainModel(Path.Combine(Directory.GetCurrentDirectory(), folderPath));
            var predictionEngine = modelBuilder.CreatePredictionEngine();
            modelBuilder.SaveModel(Path.Combine(Directory.GetCurrentDirectory(), "model.zip"));

            Console.WriteLine($"Empieza a conversar: ");
            while (true)
            {
                string inputText = Console.ReadLine();

                // Realizar una predicción
                var input = new Input { Text = inputText };
                var prediction = predictionEngine.Predict(input);

                Console.WriteLine($"Input: {inputText}");
                Console.WriteLine($"Predicted Label: {prediction.PredictedLabel}");
                Console.WriteLine($"Probability: {prediction.Probability}");
            }
        }
    }
}
