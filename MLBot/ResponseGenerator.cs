using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLBot
{
    public class ResponseGenerator
    {
        private readonly Dictionary<string, List<string>> _responses;

        public ResponseGenerator(IEnumerable<Conversation> conversations)
        {
            _responses = new Dictionary<string, List<string>>();

            foreach (var conversation in conversations)
            {
                if (!_responses.ContainsKey(conversation.Category))
                {
                    _responses[conversation.Category] = new List<string>();
                }

                _responses[conversation.Category].Add(conversation.Text);
            }
        }

        public string GenerateResponse(string category)
        {
            if (_responses.TryGetValue(category, out var responses))
            {
                var random = new Random();
                int index = random.Next(responses.Count);
                return responses[index];
            }
            else
            {
                return "I'm not sure how to respond to that.";
            }
        }
    }

}
