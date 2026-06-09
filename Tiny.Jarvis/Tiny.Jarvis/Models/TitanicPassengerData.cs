using System.Text.Json.Serialization;

namespace Tiny.Jarvis.Training.Models
{
    /// <summary>
    /// Represents a passenger record from the Titanic dataset.
    /// </summary>
    public class TitanicPassengerData
    {
        [JsonPropertyName("PassengerId")]
        public int PassengerId { get; set; }

        [JsonPropertyName("Survived")]
        public int Survived { get; set; }  // 0 = No, 1 = Yes

        [JsonPropertyName("Pclass")]
        public int Pclass { get; set; }    // 1 = 1st, 2 = 2nd, 3 = 3rd

        [JsonPropertyName("Name")]
        public string Name { get; set; } = string.Empty;

        [JsonPropertyName("Sex")]
        public string Sex { get; set; } = string.Empty;

        [JsonPropertyName("Age")]
        public double? Age { get; set; }   // nullable because some ages are missing

        [JsonPropertyName("SibSp")]
        public int SibSp { get; set; }     // number of siblings/spouses aboard

        [JsonPropertyName("Parch")]
        public int Parch { get; set; }     // number of parents/children aboard

        [JsonPropertyName("Ticket")]
        public string Ticket { get; set; } = string.Empty;

        [JsonPropertyName("Fare")]
        public double Fare { get; set; }

        [JsonPropertyName("Cabin")]
        public string Cabin { get; set; } = string.Empty;

        [JsonPropertyName("Embarked")]
        public string Embarked { get; set; } = string.Empty;  // C = Cherbourg, Q = Queenstown, S = Southampton

        public override string ToString() => $"{PassengerId}: {Name}, Class: {Pclass}, Sex: {Sex}, Age: {Age}, SibSp: {SibSp}, Parch: {Parch}, Ticket: {Ticket}, Fare: {Fare}, Cabin: {Cabin}, Embarked: {Embarked}";
    }
}
