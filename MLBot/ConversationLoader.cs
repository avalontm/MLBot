using System.Text;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace MLBot
{
    public class ConversationData
    {
        public List<string> categories { get; set; } = new List<string>();
        public List<List<string>> conversations { get; set; } = new List<List<string>>();
    }

    public class Conversation
    {
        public string? Text { get; set; }
        public string? Category { get; set; }
    }

    public static class ConversationLoader
    {
        public static IEnumerable<Conversation> LoadConversationsFromFolder(string folderPath)
        {
            var conversations = new List<Conversation>();
            var deserializer = new Deserializer();

            // Get all YAML files in the folder
            var yamlFiles = Directory.GetFiles(folderPath, "*.yml");

            foreach (var filePath in yamlFiles)
            {
                try
                {
                    var yamlText = File.ReadAllText(filePath);
                    var conversationData = deserializer.Deserialize<ConversationData>(yamlText);

                    foreach (var conversation in conversationData.conversations)
                    {
                        for (int i = 0; i < conversation.Count; i++)
                        {
                            var sentence = conversation[i];
                            var category = conversationData.categories.FirstOrDefault() ?? "Unknown";
                            conversations.Add(new Conversation { Text = sentence, Category = category });
                        }
                    }
                }catch(Exception ex)
                {
                    Console.WriteLine(ex);
                }
            }

            return conversations;
        }
    }
}

