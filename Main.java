import java.awt.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.List;
import javax.swing.*;
import javax.swing.Timer;
import javax.swing.border.EmptyBorder;

/**
 * SmartTalk - Java AI Chatbot
 *
 * Features:
 * - Java Swing graphical user interface
 * - Real-time chat interaction
 * - FAQ knowledge base
 * - Basic NLP preprocessing (lowercase, punctuation removal, stop-word removal, stemming)
 * - TF-IDF-style weighting and cosine similarity for matching questions
 * - Greeting, name memory, date/time, help, and unknown-question handling
 *
 * No external libraries are required.
 */
public class Main {
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            ChatBot bot = new ChatBot();
            new ChatWindow(bot);
        });
    }
}
class ChatBot {
    // Add more question-answer pairs here to expand the chatbot's knowledge.
    private static final String[][] FAQ = {
        {"What is Java?", "Java is a high-level, class-based, object-oriented programming language. It is commonly used to build desktop, web, and enterprise applications."},
        {"What is Python?", "Python is a high-level, interpreted programming language known for its readable syntax. It is used in automation, data science, AI, and web development."},
        {"What is artificial intelligence?", "Artificial Intelligence (AI) is a field of computer science that develops systems able to perform tasks that normally require human intelligence, such as reasoning and language understanding."},
        {"What is machine learning?", "Machine learning is a branch of AI in which computers learn patterns from data and use those patterns to make predictions or decisions."},
        {"What is deep learning?", "Deep learning is a type of machine learning that uses multi-layered neural networks to learn complex patterns from data."},
        {"What is NLP?", "Natural Language Processing (NLP) is a field of AI that helps computers process, analyse, and respond to human language."},
        {"What is an algorithm?", "An algorithm is a finite, step-by-step set of instructions used to solve a problem or complete a task."},
        {"What is a variable in programming?", "A variable is a named storage location used by a program to hold a value that can be used or changed."},
        {"What is object oriented programming?", "Object-oriented programming (OOP) organizes software around objects that contain data and methods. Its main concepts include encapsulation, inheritance, polymorphism, and abstraction."},
        {"What is a class in Java?", "A class in Java is a blueprint used to create objects. It can contain fields, constructors, and methods."},
        {"What is an object in Java?", "An object is an instance of a class. It has state, represented by fields, and behaviour, represented by methods."},
        {"What is HTML?", "HTML (HyperText Markup Language) is the standard markup language used to structure content on web pages."},
        {"What is CSS?", "CSS (Cascading Style Sheets) is used to control the appearance and layout of HTML pages."},
        {"What is JavaScript?", "JavaScript is a programming language commonly used to make web pages interactive. It can also be used on servers."},
        {"What is a database?", "A database is an organized collection of data that can be stored, searched, updated, and managed."},
        {"What is SQL?", "SQL (Structured Query Language) is used to create, read, update, and delete data in relational databases."},
        {"What is cybersecurity?", "Cybersecurity is the practice of protecting computers, networks, applications, and data from digital attacks and unauthorized access."},
        {"What is cloud computing?", "Cloud computing provides computing services such as storage, servers, and databases over the internet."},
        {"What is data structure?", "A data structure is a way of organizing and storing data so that it can be accessed and modified efficiently."},
        {"What is an array?", "An array is a data structure that stores a fixed-size sequence of elements of the same data type, accessed using indexes."},
        {"What is Git?", "Git is a version control system that tracks changes in files and helps developers collaborate on software projects."},
        {"What is GitHub?", "GitHub is an online platform for hosting Git repositories and collaborating on software development."},
        {"How can I improve my programming skills?", "Practise consistently: learn one concept, write small programs, solve problems, read error messages, and build projects."},
        {"How do I start learning AI?", "Start with Python, basic mathematics, data handling, and statistics. Then learn machine learning concepts and build small projects."},
        {"What is the difference between AI and machine learning?", "AI is the broader field of building intelligent systems. Machine learning is a subset of AI that learns patterns from data."}
    };

    private static final Set<String> STOP_WORDS = new HashSet<>(Arrays.asList(
        "a", "an", "the", "is", "are", "am", "was", "were", "be", "been",
        "do", "does", "did", "i", "me", "my", "you", "your", "we", "our",
        "to", "of", "in", "on", "for", "and", "or", "can", "could", "would",
        "what", "how", "why", "when", "where", "who", "it", "this", "that",
        "with", "at", "by", "from", "about", "please", "tell", "explain"
    ));

    private static final double MATCH_THRESHOLD = 0.20;

    private final List<Map<String, Double>> faqVectors = new ArrayList<>();
    private final Map<String, Double> inverseDocumentFrequency = new HashMap<>();
    private String userName;

    ChatBot() {
        trainFaqMatcher();
    }

    private void trainFaqMatcher() {
        Map<String, Integer> documentFrequency = new HashMap<>();
        List<List<String>> documents = new ArrayList<>();

        for (String[] item : FAQ) {
            List<String> tokens = tokenize(item[0]);
            documents.add(tokens);

            Set<String> uniqueTokens = new HashSet<>(tokens);
            for (String token : uniqueTokens) {
                Integer oldCount = documentFrequency.get(token);
                documentFrequency.put(token, oldCount == null ? 1 : oldCount + 1);
            }
        }

        int totalDocuments = FAQ.length;
        for (Map.Entry<String, Integer> entry : documentFrequency.entrySet()) {
            double idf = Math.log((totalDocuments + 1.0) / (entry.getValue() + 1.0)) + 1.0;
            inverseDocumentFrequency.put(entry.getKey(), idf);
        }

        for (List<String> document : documents) {
            faqVectors.add(vectorize(document));
        }
    }

    private List<String> tokenize(String text) {
        List<String> result = new ArrayList<>();
        String cleaned = text.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9 ]", " ");

        for (String word : cleaned.split("\\s+")) {
            if (!word.isEmpty() && !STOP_WORDS.contains(word)) {
                result.add(stem(word));
            }
        }
        return result;
    }

    // A small rule-based stemmer: reduces common word endings for better matching.
    private String stem(String word) {
        if (word.length() > 5 && word.endsWith("ing")) {
            return word.substring(0, word.length() - 3);
        }
        if (word.length() > 4 && word.endsWith("ed")) {
            return word.substring(0, word.length() - 2);
        }
        if (word.length() > 4 && word.endsWith("es")) {
            return word.substring(0, word.length() - 2);
        }
        if (word.length() > 3 && word.endsWith("s") && !word.endsWith("ss")) {
            return word.substring(0, word.length() - 1);
        }
        return word;
    }

    private Map<String, Double> vectorize(List<String> tokens) {
        Map<String, Double> vector = new HashMap<>();
        for (String token : tokens) {
            Double idf = inverseDocumentFrequency.get(token);
            if (idf != null) {
                Double oldWeight = vector.get(token);
                vector.put(token, oldWeight == null ? idf : oldWeight + idf);
            }
        }
        return vector;
    }

    private double cosineSimilarity(Map<String, Double> first, Map<String, Double> second) {
        double dotProduct = 0.0;
        double firstNorm = 0.0;
        double secondNorm = 0.0;

        for (Map.Entry<String, Double> entry : first.entrySet()) {
            double value = entry.getValue();
            firstNorm += value * value;

            Double otherValue = second.get(entry.getKey());
            if (otherValue != null) {
                dotProduct += value * otherValue;
            }
        }

        for (double value : second.values()) {
            secondNorm += value * value;
        }

        if (firstNorm == 0.0 || secondNorm == 0.0) {
            return 0.0;
        }
        return dotProduct / (Math.sqrt(firstNorm) * Math.sqrt(secondNorm));
    }

    String greeting() {
        return "Hello! I'm SmartTalk, a Java chatbot. Ask me about programming, AI, technology, or type 'help' to see examples.";
    }

    boolean isExit(String input) {
        String value = input.trim().toLowerCase(Locale.ROOT);
        return value.equals("exit") || value.equals("quit") || value.equals("bye") || value.equals("goodbye");
    }

    String getResponse(String input) {
        if (input == null || input.trim().isEmpty()) {
            return "Please type a message so I can help you.";
        }

        String lower = input.trim().toLowerCase(Locale.ROOT);
        Set<String> words = new HashSet<>(Arrays.asList(
            lower.replaceAll("[^a-z0-9' ]", " ").split("\\s+")
        ));

        if (lower.startsWith("my name is ") || lower.startsWith("call me ")) {
            String[] parts = input.trim().split("\\s+");
            String name = parts[parts.length - 1].replaceAll("[^a-zA-Z'-]", "");
            if (!name.isEmpty()) {
                userName = Character.toUpperCase(name.charAt(0)) + name.substring(1);
                return "Nice to meet you, " + userName + "! I'll remember your name during this conversation.";
            }
        }

        if (isExit(lower)) {
            return "Goodbye" + (userName == null ? "" : ", " + userName) + "! Thanks for chatting with SmartTalk.";
        }

        if (words.contains("help")) {
            return "You can ask me questions such as: 'What is Java?', 'Explain AI', 'What is machine learning?', 'What is an algorithm?', 'What is HTML?', or 'What is cybersecurity?'. You can also ask for the time or date.";
        }

        if (words.contains("hello") || words.contains("hi") || words.contains("hey")) {
            return "Hello" + (userName == null ? "" : ", " + userName) + "! What would you like to learn today?";
        }

        if (words.contains("thanks") || words.contains("thank")) {
            return "You're welcome! Feel free to ask me another question.";
        }

        if (words.contains("time")) {
            return "The current time is " +
                LocalDateTime.now().format(DateTimeFormatter.ofPattern("hh:mm a")) + ".";
        }

        if (words.contains("date") || words.contains("today")) {
            return "Today's date is " +
                LocalDateTime.now().format(DateTimeFormatter.ofPattern("EEEE, dd MMMM yyyy")) + ".";
        }

        if ((words.contains("what") || words.contains("who")) &&
                (words.contains("name") || words.contains("are")) &&
                (lower.contains("your") || lower.contains("you"))) {
            return "I'm SmartTalk, a Java chatbot built with basic NLP and FAQ matching.";
        }

        Map<String, Double> questionVector = vectorize(tokenize(lower));
        double bestScore = 0.0;
        int bestIndex = -1;

        for (int i = 0; i < faqVectors.size(); i++) {
            double score = cosineSimilarity(questionVector, faqVectors.get(i));
            if (score > bestScore) {
                bestScore = score;
                bestIndex = i;
            }
        }

        if (bestIndex >= 0 && bestScore >= MATCH_THRESHOLD) {
            return FAQ[bestIndex][1];
        }

        return "I'm not sure about that yet. Please try asking in a different way, or ask me about Java, Python, AI, machine learning, programming, or technology.";
    }
}
class ChatWindow extends JFrame {
    private final JTextArea conversationArea = new JTextArea();
    private final JTextField messageField = new JTextField();
    private final ChatBot bot;

    ChatWindow(ChatBot bot) {
        super("SmartTalk - Java AI Chatbot");
        this.bot = bot;

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(650, 600);
        setMinimumSize(new Dimension(480, 420));
        setLocationRelativeTo(null);

        JPanel mainPanel = new JPanel(new BorderLayout(10, 10));
        mainPanel.setBorder(new EmptyBorder(12, 12, 12, 12));
        mainPanel.setBackground(new Color(245, 247, 250));

        JLabel heading = new JLabel("SmartTalk  |  AI Chatbot", SwingConstants.CENTER);
        heading.setFont(new Font("SansSerif", Font.BOLD, 21));
        heading.setBorder(new EmptyBorder(8, 5, 12, 5));
        mainPanel.add(heading, BorderLayout.NORTH);

        conversationArea.setEditable(false);
        conversationArea.setLineWrap(true);
        conversationArea.setWrapStyleWord(true);
        conversationArea.setFont(new Font("SansSerif", Font.PLAIN, 15));
        conversationArea.setMargin(new Insets(12, 12, 12, 12));
        conversationArea.setBackground(Color.WHITE);

        JScrollPane scrollPane = new JScrollPane(conversationArea);
        mainPanel.add(scrollPane, BorderLayout.CENTER);

        JButton sendButton = new JButton("Send");
        JButton clearButton = new JButton("Clear Chat");

        sendButton.addActionListener(event -> sendMessage());
        messageField.addActionListener(event -> sendMessage());
        clearButton.addActionListener(event -> clearChat());

        JPanel inputPanel = new JPanel(new BorderLayout(8, 8));
        inputPanel.setBackground(new Color(245, 247, 250));
        inputPanel.add(messageField, BorderLayout.CENTER);

        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.RIGHT, 5, 0));
        buttonPanel.setBackground(new Color(245, 247, 250));
        buttonPanel.add(clearButton);
        buttonPanel.add(sendButton);
        inputPanel.add(buttonPanel, BorderLayout.EAST);

        mainPanel.add(inputPanel, BorderLayout.SOUTH);
        setContentPane(mainPanel);

        appendMessage("SmartTalk", bot.greeting());
        setVisible(true);
        messageField.requestFocusInWindow();
    }

    private void sendMessage() {
        final String message = messageField.getText().trim();
        if (message.isEmpty()) {
            return;
        }

        messageField.setText("");
        appendMessage("You", message);
        final String response = bot.getResponse(message);
        final boolean shouldExit = bot.isExit(message);

        Timer replyTimer = new Timer(350, event -> {
            appendMessage("SmartTalk", response);
            if (shouldExit) {
                Timer closeTimer = new Timer(900, closeEvent -> System.exit(0));
                closeTimer.setRepeats(false);
                closeTimer.start();
            }
        });
        replyTimer.setRepeats(false);
        replyTimer.start();
    }

    private void clearChat() {
        conversationArea.setText("");
        appendMessage("SmartTalk", "Chat cleared. What would you like to ask?");
    }

    private void appendMessage(String sender, String message) {
        conversationArea.append(sender + ": " + message + "\n\n");
        conversationArea.setCaretPosition(conversationArea.getDocument().getLength());
    }
}
