#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cstdint>

// POSIX sockets for lightweight persistent server mode
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

// ==========================================
// 1. TIFINAGH TOKENIZER ENGINE
// ==========================================
class TifinaghTokenizer {
private:
    std::unordered_map<std::string, int> char_to_id;
    std::unordered_map<int, std::string> id_to_char;

    std::vector<std::string> split_utf8(const std::string& str) const {
        std::vector<std::string> chars;
        size_t i = 0;
        while (i < str.length()) {
            unsigned char c = (unsigned char)str[i];
            size_t char_len = 1;

            if ((c & 0x80) == 0) char_len = 1;
            else if ((c & 0xE0) == 0xC0) char_len = 2;
            else if ((c & 0xF0) == 0xE0) char_len = 3;
            else if ((c & 0xF8) == 0xF0) char_len = 4;

            if (i + char_len <= str.length()) {
                chars.push_back(str.substr(i, char_len));
            }
            i += char_len;
        }
        return chars;
    }

public:
    TifinaghTokenizer() {
        std::vector<std::string> alphabet = {
            "<pad>", "<unk>", " ",
            "ⴰ", "ⴱ", "ⴳ", "ⴷ", "ⴹ", "ⴻ", "ⴼ", "ⴽ", "ⵀ", "ⵃ", "ⵄ", "ⵅ", "ⵇ", 
            "ⵉ", "ⵊ", "ⵍ", "ⵎ", "ⵏ", "ⵓ", "ⵔ", "ⵕ", "ⵖ", "ⵙ", "ⵚ", "ⵛ", "ⵜ", 
            "ⵟ", "ⵡ", "ⵢ", "ⵣ", "ⵥ", "ⴲ", "ⵯ", "ⴴ", "ⴵ",
            ".", ",", "!", "?", "-", ":", "'", "\"", "⵰"
        };
        for (size_t i = 0; i < alphabet.size(); ++i) {
            char_to_id[alphabet[i]] = (int)i;
            id_to_char[(int)i] = alphabet[i];
        }
    }

    std::vector<std::string> tokenize_words(const std::string& str) const {
        std::vector<std::string> words;
        std::string current_word = "";
        for (const auto& ch : split_utf8(str)) {
            if (ch == " " || ch == "\t" || ch == "\n" || ch == "\r" || 
                ch == "?" || ch == "." || ch == "," || ch == "!" || 
                ch == ":" || ch == "⵰" || ch == ";" || ch == "-") {
                if (!current_word.empty()) {
                    words.push_back(current_word);
                    current_word = "";
                }
            } else {
                current_word += ch;
            }
        }
        if (!current_word.empty()) words.push_back(current_word);
        return words;
    }

    std::string normalize_text(const std::string& str) const {
        std::string res = "";
        for (const auto& w : tokenize_words(str)) {
            if (!res.empty()) res += " ";
            res += w;
        }
        return res;
    }

    size_t alphabet_size() const { return char_to_id.size(); }
};

// ==========================================
// 2. MATHEMATICAL HELPERS & ACTIVATIONS
// ==========================================
inline double clip_value(double val, double min_val, double max_val) {
    if (val > max_val) return max_val;
    if (val < min_val) return min_val;
    return val;
}

inline double relu(double x) { return x > 0 ? x : 0.0; }
inline double relu_derivative(double x) { return x > 0 ? 1.0 : 0.0; }

std::vector<double> softmax(const std::vector<double>& z) {
    if (z.empty()) return {};
    std::vector<double> exp_z(z.size());
    double max_val = *std::max_element(z.begin(), z.end());
    double sum = 0.0;
    for (size_t i = 0; i < z.size(); ++i) {
        exp_z[i] = std::exp(z[i] - max_val);
        sum += exp_z[i];
    }
    if (sum <= 0.0) sum = 1e-9;
    for (size_t i = 0; i < z.size(); ++i) {
        exp_z[i] /= sum;
    }
    return exp_z;
}

// JSON escaping helper preserving UTF-8 multibyte characters
std::string escape_json(const std::string& s) {
    std::ostringstream o;
    for (size_t i = 0; i < s.length(); ++i) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
            case '"':  o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\b': o << "\\b";  break;
            case '\f': o << "\\f";  break;
            case '\n': o << "\\n";  break;
            case '\r': o << "\\r";  break;
            case '\t': o << "\\t";  break;
            default:
                if (c < 32) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    o << buf;
                } else {
                    o << (char)c;
                }
                break;
        }
    }
    return o.str();
}

std::string extract_value(const std::string& line, const std::string& key) {
    size_t key_pos = line.find(key);
    if (key_pos == std::string::npos) return "";
    size_t start = line.find("\"", key_pos + key.length());
    if (start == std::string::npos) return "";
    start++;

    size_t end = start;
    while (end < line.length()) {
        if (line[end] == '\"' && line[end - 1] != '\\') break;
        end++;
    }
    if (end >= line.length()) return "";

    std::string val = line.substr(start, end - start);
    // Unescape quotes and backslashes
    std::string unescaped = "";
    for (size_t i = 0; i < val.length(); ++i) {
        if (val[i] == '\\' && i + 1 < val.length()) {
            char next = val[i + 1];
            if (next == '\"' || next == '\\') {
                unescaped += next;
                i++;
                continue;
            } else if (next == 'n') {
                unescaped += '\n';
                i++;
                continue;
            }
        }
        unescaped += val[i];
    }
    return unescaped;
}

// ==========================================
// 3. NEURAL NETWORK MODEL STRUCTURE
// ==========================================
struct AwalModel {
    int input_size = 0;
    int hidden_size = 64;
    int output_size = 0;

    std::vector<std::string> vocab;
    std::vector<std::string> responses;

    // Fast exact / normalized prompt lookup cache
    std::unordered_map<std::string, int> exact_prompt_map;

    std::vector<std::vector<double>> W1;
    std::vector<double> B1;
    std::vector<std::vector<double>> W2;
    std::vector<double> B2;
};

// Check if file exists
bool file_exists(const std::string& path) {
    std::ifstream f(path.c_str());
    return f.good();
}

// Resolve dataset path relative to executable or current directory
std::string resolve_path(const std::string& filename, const std::string& exec_dir) {
    if (file_exists(filename)) return filename;
    if (!exec_dir.empty()) {
        std::string p = exec_dir + "/" + filename;
        if (file_exists(p)) return p;
    }
    if (file_exists("backend/" + filename)) return "backend/" + filename;
    if (file_exists("../backend/" + filename)) return "../backend/" + filename;
    return filename;
}

// Save model to binary format
bool save_model(const std::string& filepath, const AwalModel& model) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) return false;

    const char magic[8] = {'A', 'W', 'A', 'L', '0', '0', '0', '2'};
    out.write(magic, 8);

    int32_t hs = model.hidden_size;
    int32_t is = model.input_size;
    int32_t os = model.output_size;

    out.write(reinterpret_cast<const char*>(&hs), sizeof(hs));
    out.write(reinterpret_cast<const char*>(&is), sizeof(is));
    out.write(reinterpret_cast<const char*>(&os), sizeof(os));

    // Vocab
    uint32_t vocab_count = model.vocab.size();
    out.write(reinterpret_cast<const char*>(&vocab_count), sizeof(vocab_count));
    for (const auto& w : model.vocab) {
        uint32_t len = w.size();
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(w.data(), len);
    }

    // Responses
    uint32_t resp_count = model.responses.size();
    out.write(reinterpret_cast<const char*>(&resp_count), sizeof(resp_count));
    for (const auto& r : model.responses) {
        uint32_t len = r.size();
        out.write(reinterpret_cast<const char*>(&len), sizeof(len));
        out.write(r.data(), len);
    }

    // Exact prompt map
    uint32_t prompt_map_count = model.exact_prompt_map.size();
    out.write(reinterpret_cast<const char*>(&prompt_map_count), sizeof(prompt_map_count));
    for (const auto& pair : model.exact_prompt_map) {
        uint32_t klen = pair.first.size();
        out.write(reinterpret_cast<const char*>(&klen), sizeof(klen));
        out.write(pair.first.data(), klen);
        int32_t val = pair.second;
        out.write(reinterpret_cast<const char*>(&val), sizeof(val));
    }

    // W1
    for (int i = 0; i < hs; ++i) {
        out.write(reinterpret_cast<const char*>(model.W1[i].data()), is * sizeof(double));
    }
    // B1
    out.write(reinterpret_cast<const char*>(model.B1.data()), hs * sizeof(double));

    // W2
    for (int i = 0; i < os; ++i) {
        out.write(reinterpret_cast<const char*>(model.W2[i].data()), hs * sizeof(double));
    }
    // B2
    out.write(reinterpret_cast<const char*>(model.B2.data()), os * sizeof(double));

    out.close();
    return true;
}

// Load model from binary format
bool load_model(const std::string& filepath, AwalModel& model) {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open()) return false;

    char magic[8];
    in.read(magic, 8);
    if (std::memcmp(magic, "AWAL0002", 8) != 0 && std::memcmp(magic, "AWAL0001", 8) != 0) {
        return false;
    }
    bool is_v2 = (std::memcmp(magic, "AWAL0002", 8) == 0);

    int32_t hs, is, os;
    in.read(reinterpret_cast<char*>(&hs), sizeof(hs));
    in.read(reinterpret_cast<char*>(&is), sizeof(is));
    in.read(reinterpret_cast<char*>(&os), sizeof(os));

    model.hidden_size = hs;
    model.input_size = is;
    model.output_size = os;

    // Vocab
    uint32_t vocab_count;
    in.read(reinterpret_cast<char*>(&vocab_count), sizeof(vocab_count));
    model.vocab.resize(vocab_count);
    for (uint32_t i = 0; i < vocab_count; ++i) {
        uint32_t len;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string w(len, '\0');
        in.read(&w[0], len);
        model.vocab[i] = w;
    }

    // Responses
    uint32_t resp_count;
    in.read(reinterpret_cast<char*>(&resp_count), sizeof(resp_count));
    model.responses.resize(resp_count);
    for (uint32_t i = 0; i < resp_count; ++i) {
        uint32_t len;
        in.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string r(len, '\0');
        in.read(&r[0], len);
        model.responses[i] = r;
    }

    // Exact prompt map (if v2)
    model.exact_prompt_map.clear();
    if (is_v2) {
        uint32_t prompt_map_count;
        in.read(reinterpret_cast<char*>(&prompt_map_count), sizeof(prompt_map_count));
        for (uint32_t i = 0; i < prompt_map_count; ++i) {
            uint32_t klen;
            in.read(reinterpret_cast<char*>(&klen), sizeof(klen));
            std::string k(klen, '\0');
            in.read(&k[0], klen);
            int32_t val;
            in.read(reinterpret_cast<char*>(&val), sizeof(val));
            model.exact_prompt_map[k] = val;
        }
    }

    // W1
    model.W1.assign(hs, std::vector<double>(is, 0.0));
    for (int i = 0; i < hs; ++i) {
        in.read(reinterpret_cast<char*>(model.W1[i].data()), is * sizeof(double));
    }
    // B1
    model.B1.resize(hs);
    in.read(reinterpret_cast<char*>(model.B1.data()), hs * sizeof(double));

    // W2
    model.W2.assign(os, std::vector<double>(hs, 0.0));
    for (int i = 0; i < os; ++i) {
        in.read(reinterpret_cast<char*>(model.W2[i].data()), hs * sizeof(double));
    }
    // B2
    model.B2.resize(os);
    in.read(reinterpret_cast<char*>(model.B2.data()), os * sizeof(double));

    in.close();
    return true;
}

// Forward inference
std::pair<std::string, int> predict(const AwalModel& model, const TifinaghTokenizer& tokenizer, const std::string& input_text) {
    if (input_text.empty() || model.responses.empty()) {
        return {"ⴰⵣⵓⵍ! ⵎⴰⵏⵉⴽ ⵙ ⴽ ⵉⵣⴹⴰⵔⵖ ⴰⴷ ⴰⵡⵙⵖ?", 50};
    }

    // Check exact / normalized prompt lookup first (high precision retrieval)
    std::string norm_input = tokenizer.normalize_text(input_text);
    auto direct_it = model.exact_prompt_map.find(input_text);
    if (direct_it != model.exact_prompt_map.end() && direct_it->second < (int)model.responses.size()) {
        return {model.responses[direct_it->second], 98};
    }
    auto norm_it = model.exact_prompt_map.find(norm_input);
    if (norm_it != model.exact_prompt_map.end() && norm_it->second < (int)model.responses.size()) {
        return {model.responses[norm_it->second], 96};
    }

    // Tokenize words for neural network forward pass
    std::vector<std::string> words = tokenizer.tokenize_words(input_text);
    std::vector<int> w_indices;
    for (const auto& w : words) {
        auto it = std::find(model.vocab.begin(), model.vocab.end(), w);
        if (it != model.vocab.end()) {
            w_indices.push_back(std::distance(model.vocab.begin(), it));
        }
    }

    // Fallback if no vocabulary words matched
    if (w_indices.empty()) {
        return {"ⴰⵣⵓⵍ! ⵓⵔ ⴳⵣⵉⵖ ⵖⵉⴽⴰⵏ, ⵎⴰⵏⵉⴽ ⵙ ⴽ ⵉⵣⴹⴰⵔⵖ ⴰⴷ ⴰⵡⵙⵖ? (Azul! How can I assist you?)", 50};
    }

    // Forward Pass: Layer 1
    std::vector<double> h(model.hidden_size, 0.0);
    for (int i = 0; i < model.hidden_size; ++i) {
        double sum = model.B1[i];
        for (int wid : w_indices) {
            sum += model.W1[i][wid];
        }
        h[i] = relu(sum);
    }

    // Forward Pass: Layer 2
    std::vector<double> z(model.output_size, 0.0);
    for (int i = 0; i < model.output_size; ++i) {
        double sum = model.B2[i];
        for (int j = 0; j < model.hidden_size; ++j) {
            sum += model.W2[i][j] * h[j];
        }
        z[i] = sum;
    }

    std::vector<double> probs = softmax(z);
    int best_idx = 0;
    double max_prob = -1.0;
    for (size_t i = 0; i < probs.size(); ++i) {
        if (probs[i] > max_prob) {
            max_prob = probs[i];
            best_idx = i;
        }
    }

    // Confidence calibration:
    // With N = 897 classes, uniform baseline is 1/N = ~0.0011 (0.11%).
    // Scale relative probability into [50..97]% for clear user display
    double baseline = 1.0 / (double)std::max(1, model.output_size);
    double ratio = max_prob / (baseline + 1e-9);
    double score = 55.0 + (40.0 * (1.0 - std::exp(-std::max(0.0, ratio - 1.0) / 7.0)));
    int confidence = (int)std::round(score);
    if (confidence < 50) confidence = 50;
    if (confidence > 97) confidence = 97;

    return {model.responses[best_idx], confidence};
}

// Train model from dataset.jsonl
bool train_model(const std::string& dataset_path, const std::string& weights_path, AwalModel& model, TifinaghTokenizer& tokenizer, bool quiet = false) {
    srand(1337);

    std::ifstream file(dataset_path);
    if (!file.is_open()) {
        std::cerr << "Error: dataset file not found at: " << dataset_path << std::endl;
        std::cerr << "Please provide a valid 'dataset.jsonl' or get the curated Tamazight NLP dataset at:\n"
                  << "👉 https://rachidnichan.com/l/Tamazight-NLP-Dataset\n" << std::endl;
        return false;
    }

    std::vector<std::pair<std::vector<int>, int>> train_data;
    model.vocab.clear();
    model.responses.clear();
    model.exact_prompt_map.clear();

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        size_t user_idx = line.find("\"role\": \"user\"");
        size_t bot_idx = line.find("\"role\": \"assistant\"");

        if (user_idx != std::string::npos && bot_idx != std::string::npos) {
            std::string q = extract_value(line.substr(user_idx), "\"content\":");
            std::string a = extract_value(line.substr(bot_idx), "\"content\":");

            if (!q.empty() && !a.empty()) {
                auto resp_it = std::find(model.responses.begin(), model.responses.end(), a);
                int target_id;
                if (resp_it == model.responses.end()) {
                    target_id = model.responses.size();
                    model.responses.push_back(a);
                } else {
                    target_id = std::distance(model.responses.begin(), resp_it);
                }

                // Add to prompt map for instant exact retrieval
                model.exact_prompt_map[q] = target_id;
                std::string norm_q = tokenizer.normalize_text(q);
                if (!norm_q.empty()) {
                    model.exact_prompt_map[norm_q] = target_id;
                }

                std::vector<int> word_ids;
                for (const auto& w : tokenizer.tokenize_words(q)) {
                    auto vocab_it = std::find(model.vocab.begin(), model.vocab.end(), w);
                    int wid;
                    if (vocab_it == model.vocab.end()) {
                        wid = model.vocab.size();
                        model.vocab.push_back(w);
                    } else {
                        wid = std::distance(model.vocab.begin(), vocab_it);
                    }
                    word_ids.push_back(wid);
                }
                train_data.push_back({word_ids, target_id});
            }
        }
    }
    file.close();

    model.input_size = model.vocab.size();
    model.hidden_size = 64;
    model.output_size = model.responses.size();

    if (model.input_size == 0 || model.output_size == 0) {
        std::cerr << "Error: No training samples loaded from " << dataset_path << std::endl;
        return false;
    }

    if (!quiet) {
        std::cout << "Dataset: " << train_data.size() << " samples | Vocab: " << model.input_size << " words | Classes: " << model.output_size << std::endl;
        std::cout << "--- Training Neural Network with Gradient Safety ---" << std::endl;
    }

    // Initialize Weights
    model.W1.assign(model.hidden_size, std::vector<double>(model.input_size, 0.0));
    model.B1.assign(model.hidden_size, 0.0);
    model.W2.assign(model.output_size, std::vector<double>(model.hidden_size, 0.0));
    model.B2.assign(model.output_size, 0.0);

    for (int i = 0; i < model.hidden_size; ++i)
        for (int j = 0; j < model.input_size; ++j)
            model.W1[i][j] = ((double)rand() / RAND_MAX - 0.5) * 0.05;

    for (int i = 0; i < model.output_size; ++i)
        for (int j = 0; j < model.hidden_size; ++j)
            model.W2[i][j] = ((double)rand() / RAND_MAX - 0.5) * 0.05;

    auto best_W1 = model.W1;
    auto best_B1 = model.B1;
    auto best_W2 = model.W2;
    auto best_B2 = model.B2;
    double min_recorded_loss = 9999.0;

    double base_lr = 0.0035;
    int epochs = 160;

    for (int epoch = 0; epoch <= epochs; ++epoch) {
        double total_loss = 0.0;
        double current_lr = base_lr / (1.0 + 0.004 * epoch);

        for (const auto& item : train_data) {
            const auto& w_indices = item.first;
            int target = item.second;

            // Forward Pass
            std::vector<double> h(model.hidden_size, 0.0);
            for (int i = 0; i < model.hidden_size; ++i) {
                double sum = model.B1[i];
                for (int wid : w_indices) sum += model.W1[i][wid];
                h[i] = relu(sum);
            }

            std::vector<double> z(model.output_size, 0.0);
            for (int i = 0; i < model.output_size; ++i) {
                double sum = model.B2[i];
                for (int j = 0; j < model.hidden_size; ++j) sum += model.W2[i][j] * h[j];
                z[i] = sum;
            }

            std::vector<double> probs = softmax(z);
            total_loss += -std::log(probs[target] + 1e-9);

            // Backward Pass with Gradient Clipping
            std::vector<double> dz(model.output_size);
            for (int i = 0; i < model.output_size; ++i) {
                double raw_grad = probs[i] - (i == target ? 1.0 : 0.0);
                dz[i] = clip_value(raw_grad, -1.0, 1.0);
            }

            for (int i = 0; i < model.output_size; ++i) {
                for (int j = 0; j < model.hidden_size; ++j) model.W2[i][j] -= current_lr * dz[i] * h[j];
                model.B2[i] -= current_lr * dz[i];
            }

            std::vector<double> dh(model.hidden_size, 0.0);
            for (int j = 0; j < model.hidden_size; ++j) {
                for (int i = 0; i < model.output_size; ++i) dh[j] += model.W2[i][j] * dz[i];
                dh[j] = clip_value(dh[j] * relu_derivative(h[j]), -1.0, 1.0);
            }

            for (int i = 0; i < model.hidden_size; ++i) {
                for (int wid : w_indices) model.W1[i][wid] -= current_lr * dh[i];
                model.B1[i] -= current_lr * dh[i];
            }
        }

        double avg_loss = total_loss / train_data.size();

        if (avg_loss < min_recorded_loss) {
            min_recorded_loss = avg_loss;
            best_W1 = model.W1; best_B1 = model.B1;
            best_W2 = model.W2; best_B2 = model.B2;
        }

        if (!quiet && epoch % 15 == 0) {
            std::cout << "Epoch: " << epoch << " / " << epochs 
                      << " | Loss: " << avg_loss 
                      << " | Best: " << min_recorded_loss << std::endl;
        }

        if (epoch > 50 && avg_loss > min_recorded_loss + 0.6) {
            if (!quiet) std::cout << ">>> Early Stopping Triggered! Restoring optimal brain weights. <<<" << std::endl;
            break;
        }
    }

    model.W1 = best_W1; model.B1 = best_B1;
    model.W2 = best_W2; model.B2 = best_B2;

    if (!quiet) {
        std::cout << "--- Training Complete! Optimal Brain Restored (Loss: " << min_recorded_loss << ") ---" << std::endl;
    }

    // Save model to binary file
    save_model(weights_path, model);
    if (!quiet) {
        std::cout << "Model cached to " << weights_path << " for instant loading." << std::endl;
    }

    return true;
}

// Append new conversation to dataset.jsonl
bool append_learning(const std::string& dataset_path, const std::string& user_q, const std::string& assistant_a) {
    std::ofstream out(dataset_path, std::ios::app);
    if (!out.is_open()) return false;
    out << "{\"messages\": [{\"role\": \"user\", \"content\": \"" 
        << escape_json(user_q) << "\"}, {\"role\": \"assistant\", \"content\": \"" 
        << escape_json(assistant_a) << "\"}]}\n";
    out.close();
    return true;
}

// Micro HTTP Server for Persistent Daemon Mode
void run_http_server(int port, AwalModel& model, const TifinaghTokenizer& tokenizer, const std::string& dataset_path) {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        std::cerr << "Failed to create socket on port " << port << std::endl;
        return;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in address;
    std::memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK); // strictly listen on 127.0.0.1 (loopback only)
    address.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        std::cerr << "Failed to bind to port " << port << std::endl;
        close(server_fd);
        return;
    }

    if (listen(server_fd, 64) < 0) {
        std::cerr << "Failed to listen on socket" << std::endl;
        close(server_fd);
        return;
    }

    std::cout << "🚀 Awal LM C++ Micro Server running on http://127.0.0.1:" << port << std::endl;
    std::cout << "Endpoints:" << std::endl;
    std::cout << "  - POST /predict (JSON: {\"prompt\": \"...\"} or {\"message\": \"...\"})" << std::endl;
    std::cout << "  - POST /learn   (JSON: {\"user_prompt\": \"...\", \"assistant_response\": \"...\"})" << std::endl;
    std::cout << "  - GET  /health" << std::endl;

    while (true) {
        sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) continue;

        char buffer[16384];
        ssize_t bytes_read = read(client_fd, buffer, sizeof(buffer) - 1);
        if (bytes_read <= 0) {
            close(client_fd);
            continue;
        }
        buffer[bytes_read] = '\0';
        std::string req(buffer, bytes_read);

        // Parse HTTP Method & Path
        std::string method = "GET";
        std::string path = "/";
        size_t first_space = req.find(' ');
        if (first_space != std::string::npos) {
            method = req.substr(0, first_space);
            size_t second_space = req.find(' ', first_space + 1);
            if (second_space != std::string::npos) {
                path = req.substr(first_space + 1, second_space - first_space - 1);
            }
        }

        // Find body
        std::string body = "";
        size_t header_end = req.find("\r\n\r\n");
        if (header_end != std::string::npos) {
            body = req.substr(header_end + 4);
        }

        std::string response_payload = "";
        int status_code = 200;

        if (path == "/health" && method == "GET") {
            response_payload = "{\"status\":\"ok\",\"model\":\"awal-lm\",\"vocab_size\":" + 
                               std::to_string(model.input_size) + ",\"classes\":" + 
                               std::to_string(model.output_size) + "}";
        } else if ((path == "/predict" || path == "/api/chat" || path == "/generate") && method == "POST") {
            std::string prompt = extract_value(body, "\"prompt\":");
            if (prompt.empty()) prompt = extract_value(body, "\"message\":");
            if (prompt.empty()) prompt = extract_value(body, "\"content\":");

            auto [reply, confidence] = predict(model, tokenizer, prompt);
            response_payload = "{\"response\":\"" + escape_json(reply) + 
                               "\",\"confidence\":" + std::to_string(confidence) + 
                               ",\"prompt\":\"" + escape_json(prompt) + "\"}";
        } else if (path == "/learn" && method == "POST") {
            std::string u_prompt = extract_value(body, "\"user_prompt\":");
            std::string a_resp = extract_value(body, "\"assistant_response\":");
            if (!u_prompt.empty() && !a_resp.empty()) {
                append_learning(dataset_path, u_prompt, a_resp);
                // Also add to in-memory prompt map immediately
                auto resp_it = std::find(model.responses.begin(), model.responses.end(), a_resp);
                int target_id;
                if (resp_it == model.responses.end()) {
                    target_id = model.responses.size();
                    model.responses.push_back(a_resp);
                } else {
                    target_id = std::distance(model.responses.begin(), resp_it);
                }
                model.exact_prompt_map[u_prompt] = target_id;
                model.exact_prompt_map[tokenizer.normalize_text(u_prompt)] = target_id;
                response_payload = "{\"success\":true}";
            } else {
                status_code = 400;
                response_payload = "{\"error\":\"Missing user_prompt or assistant_response\"}";
            }
        } else {
            status_code = 404;
            response_payload = "{\"error\":\"Not Found\"}";
        }

        std::ostringstream http_out;
        http_out << "HTTP/1.1 " << status_code << " OK\r\n"
                 << "Content-Type: application/json; charset=utf-8\r\n"
                 << "Access-Control-Allow-Origin: *\r\n"
                 << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                 << "Access-Control-Allow-Headers: Content-Type\r\n"
                 << "Content-Length: " << response_payload.length() << "\r\n"
                 << "Connection: close\r\n\r\n"
                 << response_payload;

        std::string out_str = http_out.str();
        write(client_fd, out_str.data(), out_str.length());
        close(client_fd);
    }

    close(server_fd);
}

// ==========================================
// 4. MAIN ENTRY POINT & CLI HANDLERS
// ==========================================
int main(int argc, char* argv[]) {
    TifinaghTokenizer tokenizer;

    // Detect execution directory
    std::string exec_path = argv[0];
    std::string exec_dir = "";
    size_t last_slash = exec_path.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        exec_dir = exec_path.substr(0, last_slash);
    }

    std::string dataset_path = resolve_path("dataset.jsonl", exec_dir);
    std::string model_path = (!exec_dir.empty()) ? (exec_dir + "/awal_model.bin") : "awal_model.bin";

    bool force_train = false;
    bool predict_mode = false;
    bool stdin_mode = false;
    bool server_mode = false;
    int server_port = 5005;
    std::string predict_query = "";

    // Parse CLI arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--train" || arg == "-t") {
            force_train = true;
        } else if (arg == "--predict" || arg == "-p") {
            predict_mode = true;
            if (i + 1 < argc) {
                predict_query = argv[++i];
            }
        } else if (arg == "--stdin" || arg == "--json-stdin") {
            stdin_mode = true;
        } else if (arg == "--server" || arg == "-s" || arg == "--daemon") {
            server_mode = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                server_port = std::atoi(argv[++i]);
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Awal LM: Pure C++ Amazigh (Tifinagh) Neural Engine\n\n"
                      << "Usage:\n"
                      << "  " << argv[0] << " [options]\n\n"
                      << "Options:\n"
                      << "  --predict, -p <query>    Run single prediction and output JSON to stdout\n"
                      << "  --server, -s [port]      Start persistent lightweight HTTP server (default: 5005)\n"
                      << "  --stdin, --json-stdin    Read JSON or text queries from stdin and return JSON\n"
                      << "  --train, -t              Retrain model from dataset.jsonl and save to awal_model.bin\n"
                      << "  --help, -h               Show this help message\n";
            return 0;
        } else if (predict_query.empty() && !predict_mode && arg[0] != '-') {
            // Positional query: ./my_ai "ⴰⵣⵓⵍ"
            predict_mode = true;
            predict_query = arg;
        }
    }

    AwalModel model;

    // Load or Train Model
    bool model_loaded = false;
    if (!force_train && file_exists(model_path)) {
        model_loaded = load_model(model_path, model);
    }

    if (!model_loaded) {
        bool quiet_train = predict_mode || stdin_mode;
        if (!quiet_train) {
            std::cout << "==================================================" << std::endl;
            std::cout << "   AWAL-LM: TIFINAGH AI ENGINE (C++ FROM SCRATCH) " << std::endl;
            std::cout << "==================================================" << std::endl;
            std::cout << "Tokenizer initialized with " << tokenizer.alphabet_size() << " symbols." << std::endl;
            std::cout << "Loading dataset & training model..." << std::endl;
        }
        if (!train_model(dataset_path, model_path, model, tokenizer, quiet_train)) {
            std::cerr << "Failed to train or load model!" << std::endl;
            return 1;
        }
    }

    // If retraining was requested directly via --train or -t, exit cleanly!
    if (force_train && !predict_mode && !stdin_mode && !server_mode) {
        std::cout << "\nRetraining successfully finished. Model saved to " << model_path << "\n";
        return 0;
    }

    // 1. Single CLI Prediction mode
    if (predict_mode) {
        auto [reply, confidence] = predict(model, tokenizer, predict_query);
        std::cout << "{\"response\":\"" << escape_json(reply) 
                  << "\",\"confidence\":" << confidence 
                  << ",\"prompt\":\"" << escape_json(predict_query) << "\"}" 
                  << std::endl;
        return 0;
    }

    // 2. STDIN JSON mode
    if (stdin_mode) {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            std::string q = extract_value(line, "\"prompt\":");
            if (q.empty()) q = extract_value(line, "\"message\":");
            if (q.empty()) q = line; // Plain text fallback

            auto [reply, confidence] = predict(model, tokenizer, q);
            std::cout << "{\"response\":\"" << escape_json(reply) 
                      << "\",\"confidence\":" << confidence 
                      << ",\"prompt\":\"" << escape_json(q) << "\"}" 
                      << std::endl;
        }
        return 0;
    }

    // 3. Persistent HTTP Server Mode
    if (server_mode) {
        run_http_server(server_port, model, tokenizer, dataset_path);
        return 0;
    }

    // 4. Interactive Terminal Mode
    std::cout << "==================================================" << std::endl;
    std::cout << "   AWAL-LM: TIFINAGH AI ENGINE (C++ FROM SCRATCH) " << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "Model loaded! Vocab: " << model.input_size << " words | Classes: " << model.output_size << std::endl;
    std::cout << "Ready for Tifinagh conversation. Type 'exit' to quit.\n" << std::endl;

    std::string user_input;
    while (true) {
        std::cout << "\nYou (Tifinagh): ";
        if (!std::getline(std::cin, user_input)) break;
        if (user_input == "exit" || user_input == "quit") break;
        if (user_input.empty()) continue;

        auto [reply, confidence] = predict(model, tokenizer, user_input);
        std::cout << "AI (" << confidence << "%): " << reply << std::endl;
    }

    return 0;
}