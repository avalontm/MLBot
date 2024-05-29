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

            IEnumerable<Conversation> conversations = ConversationLoader.LoadConversationsFromFolder(Path.Combine(Directory.GetCurrentDirectory(), folderPath));
            var responseGenerator = new ResponseGenerator(conversations);

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
                var input = new Input { Text = userInput };
                var output = predictionEngine.Predict(input);

                // Generate and display response
                var response = responseGenerator.GenerateResponse(output.PredictedLabel);
                Console.WriteLine($"Bot: {response}");
            }
        }
    }
}
