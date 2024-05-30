using System.Text;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace MLBot
{
    public class ConversationData
    {
        public List<string>? categories { get; set; }
        public List<List<string>>? conversations { get; set; }
    }

    public class ConversationItem
    {
        public string Category { get; set; }
        public string Question { get; set; }
        public List<string> Answers { get; set; }
    }

    public static class ConversationLoader
    {
        public static List<ConversationItem> LoadConversationsFromFolder(string folderPath)
        {
            var conversationGroups = new List<ConversationItem>();
            var deserializer = new DeserializerBuilder().Build();

            // Get all YAML files in the folder
            var yamlFiles = Directory.GetFiles(folderPath, "*.yml");

            foreach (var filePath in yamlFiles)
            {
                try
                {
                    var yamlText = File.ReadAllText(filePath);
                    var conversationData = deserializer.Deserialize<ConversationData>(yamlText);

                    foreach (var category in conversationData.categories)
                    {
                        foreach (var conversation in conversationData.conversations)
                        {
                            var conversationItem = new ConversationItem
                            {
                                Category = category,
                                Question = conversation[0],
                                Answers = conversation.Skip(1).ToList()
                            };

                            conversationGroups.Add(conversationItem);
                        }
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine(ex);
                }
            }

            return conversationGroups;
        }
    }
}

