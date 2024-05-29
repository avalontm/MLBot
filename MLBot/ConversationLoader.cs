using System.Text;
using YamlDotNet.Serialization;
using YamlDotNet.Serialization.NamingConventions;

namespace MLBot
{
    public static class ConversationLoader
    {
        public static IEnumerable<Conversation> LoadConversationsFromFolder(string folderPath)
        {
            List<Conversation> conversations = new List<Conversation>();

            // Obtener la lista de archivos YAML en la carpeta
            var yamlFiles = Directory.GetFiles(folderPath, "*.yml");

            // Iterar sobre cada archivo YAML
            foreach (var yamlFile in yamlFiles)
            {
                try
                {
                    // Leer el contenido del archivo YAML
                    string yamlContent = File.ReadAllText(yamlFile);

                    // Deserializar el contenido YAML a un objeto C#
                    var deserializer = new DeserializerBuilder().Build();
                    var conversationData = deserializer.Deserialize<ConversationData>(yamlContent);

                    // Crear una conversación a partir de los datos deserializados
                    var conversation = new Conversation
                    {
                        categories = conversationData.categories,
                        conversations = conversationData.conversations
                    };

                    conversations.Add(conversation);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error al procesar el archivo {yamlFile}: {ex.Message}");
                }
            }

            return conversations;
        }
    }

    public class ConversationData
    {
        public List<string> categories { get; set; } = new List<string>();
        public List<List<string>> conversations { get; set; } = new List<List<string>>();
    }

    public class Conversation
    {
        public List<string> categories { get; set; } = new List<string>();
        public List<List<string>> conversations { get; set; } = new List<List<string>>();
    }
}

