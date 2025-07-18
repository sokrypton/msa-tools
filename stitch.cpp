#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

// Compile with OpenMP: g++ -O3 -march=native -std=c++17 -fopenmp stitch_opt.cpp -o stitch_opt
// Without OpenMP: g++ -O3 -march=native -std=c++17 stitch_opt.cpp -o stitch_opt

struct Sequence {
    std::string header;
    std::string seq;       // Original sequence
    std::string clean;     // Cleaned sequence
    std::string uid;       // UniProt ID
    uint64_t num = 0;      // UniProt number
    float coverage = 0.0f;
    float identity = 0.0f;
    bool has_uid = false;
};

// Simplified UniProt converter that matches Python exactly
class UniProtConverter {
private:
    // Use vectors for dynamic sizing to avoid array bounds issues
    std::vector<int> pa_table;
    std::vector<std::vector<std::vector<int>>> ma_table;
    std::vector<std::vector<int>> base_sizes;
    
public:
    UniProtConverter() : pa_table(256, 0), 
                        ma_table(2, std::vector<std::vector<int>>(6, std::vector<int>(256, -1))),
                        base_sizes(2, std::vector<int>(6, 0)) {
        
        // PA table: O,P,Q = 1, others = 0
        for (int c = 'A'; c <= 'Z'; c++) pa_table[c] = 0;
        pa_table['O'] = pa_table['P'] = pa_table['Q'] = 1;
        
        // Position 0 and 4: digits only
        for (int p = 0; p < 2; p++) {
            for (int pos : {0, 4}) {
                for (int c = '0'; c <= '9'; c++) {
                    ma_table[p][pos][c] = c - '0';
                }
                base_sizes[p][pos] = 10;
            }
        }
        
        // Position 1 and 2: letters + digits  
        for (int p = 0; p < 2; p++) {
            for (int pos : {1, 2}) {
                int idx = 0;
                // Letters A-Z (0-25)
                for (int c = 'A'; c <= 'Z'; c++) {
                    ma_table[p][pos][c] = idx++;
                }
                // Digits 0-9 (26-35)
                for (int c = '0'; c <= '9'; c++) {
                    ma_table[p][pos][c] = idx++;
                }
                base_sizes[p][pos] = 36;
            }
        }
        
        // Position 3: different for p=0 vs p=1
        // p=0: letters only
        int idx = 0;
        for (int c = 'A'; c <= 'Z'; c++) {
            ma_table[0][3][c] = idx++;
        }
        base_sizes[0][3] = 26;
        
        // p=1: letters + digits
        idx = 0;
        for (int c = 'A'; c <= 'Z'; c++) {
            ma_table[1][3][c] = idx++;
        }
        for (int c = '0'; c <= '9'; c++) {
            ma_table[1][3][c] = idx++;
        }
        base_sizes[1][3] = 36;
        
        // Position 5: letters only
        for (int p = 0; p < 2; p++) {
            idx = 0;
            for (int c = 'A'; c <= 'Z'; c++) {
                ma_table[p][5][c] = idx++;
            }
            base_sizes[p][5] = 26;
        }
    }
    
    uint64_t convert(const std::string& uid) const {
        if (uid.empty() || !isalpha(uid[0])) return 0;
        
        int p = pa_table[(unsigned char)uid[0]];
        uint64_t tot = 1;
        uint64_t num = 0;
        
        // Handle 10-character IDs - last 4 positions
        if (uid.length() == 10) {
            for (int n = 0; n < 4; n++) {
                int pos = 9 - n;  // 9, 8, 7, 6
                int val = ma_table[p][n][(unsigned char)uid[pos]];
                if (val >= 0) {
                    num += val * tot;
                    tot *= base_sizes[p][n];
                }
            }
        }
        
        // Process first 6 characters
        for (int n = 0; n < 6; n++) {
            int pos = 5 - n;  // 5, 4, 3, 2, 1, 0
            if (pos >= 0 && pos < (int)uid.length()) {
                int val = ma_table[p][n][(unsigned char)uid[pos]];
                if (val >= 0) {
                    num += val * tot;
                    tot *= base_sizes[p][n];
                }
            }
        }
        
        return num;
    }
};

// Global converter instance
static UniProtConverter g_converter;

// Extract UniProt ID from header
std::string extract_uniprot_id(const std::string& header) {
    size_t pos = header.find("UniRef");
    if (pos == std::string::npos) return "";
    
    size_t start = header.find('_', pos);
    if (start == std::string::npos) return "";
    start++;
    
    size_t end = start;
    while (end < header.length() && 
           header[end] != ' ' && 
           header[end] != '_' && 
           header[end] != '\t') {
        end++;
    }
    
    std::string uid = header.substr(start, end - start);
    
    // Validate
    if (uid.find("UPI") != std::string::npos) return "";
    if (uid.length() != 6 && uid.length() != 10) return "";
    if (!isalpha(uid[0])) return "";
    
    return uid;
}

// Clean sequence (remove lowercase)
std::string clean_sequence(const std::string& seq) {
    std::string result;
    result.reserve(seq.length());
    
    for (char c : seq) {
        if (!islower(c)) {
            result.push_back(c);
        }
    }
    
    return result;
}

// Read and process a3m file
std::vector<Sequence> read_and_process_a3m(const std::string& filename,
                                           float min_coverage,
                                           float min_identity) {
    std::vector<Sequence> sequences;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open " << filename << std::endl;
        return sequences;
    }
    
    // Read file
    std::string line;
    Sequence current;
    bool in_seq = false;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        if (line[0] == '>') {
            if (in_seq) {
                sequences.push_back(std::move(current));
            }
            current = Sequence();
            current.header = line;
            in_seq = true;
        } else if (in_seq) {
            current.seq += line;
        }
    }
    
    if (in_seq) {
        sequences.push_back(std::move(current));
    }
    
    file.close();
    
    if (sequences.empty()) return sequences;
    
    // Process sequences
    sequences[0].clean = clean_sequence(sequences[0].seq);
    const std::string& query = sequences[0].clean;
    const size_t query_len = query.length();
    
    // Process all sequences
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < sequences.size(); i++) {
        auto& seq = sequences[i];
        
        // Extract UniProt ID
        seq.uid = extract_uniprot_id(seq.header);
        seq.has_uid = !seq.uid.empty();
        if (seq.has_uid) {
            seq.num = g_converter.convert(seq.uid);
        }
        
        // Clean sequence
        if (i > 0) {  // Query already cleaned
            seq.clean = clean_sequence(seq.seq);
        }
        
        // Calculate metrics
        if (seq.clean.length() == query_len) {
            int gaps = 0;
            int matches = 0;
            
            for (size_t j = 0; j < query_len; j++) {
                if (seq.clean[j] == '-') {
                    gaps++;
                } else if (seq.clean[j] == query[j]) {
                    matches++;
                }
            }
            
            seq.coverage = 1.0f - (float)gaps / query_len;
            seq.identity = (float)matches / query_len;
        }
    }
    
    // Filter sequences (keep query)
    std::vector<Sequence> filtered;
    filtered.reserve(sequences.size());
    filtered.push_back(std::move(sequences[0]));  // Always keep query
    
    for (size_t i = 1; i < sequences.size(); i++) {
        if (sequences[i].has_uid && 
            sequences[i].coverage >= min_coverage && 
            sequences[i].identity >= min_identity &&
            sequences[i].clean.length() == query_len) {
            filtered.push_back(std::move(sequences[i]));
        }
    }
    
    return filtered;
}

// Optimized stitching algorithm
std::vector<std::vector<size_t>> stitch_sequences(
    const std::vector<std::vector<Sequence>>& chains,
    uint64_t max_distance) {
    
    std::vector<std::vector<size_t>> results;
    if (chains.empty() || chains[0].empty()) return results;
    
    const size_t num_chains = chains.size();
    
    // Add query stitch
    std::vector<size_t> query(num_chains, 0);
    results.push_back(query);
    
    // Create sorted indices for each chain
    std::vector<std::vector<std::pair<uint64_t, size_t>>> sorted_chains(num_chains);
    
    for (size_t c = 0; c < num_chains; c++) {
        for (size_t i = 1; i < chains[c].size(); i++) {
            if (chains[c][i].has_uid) {
                sorted_chains[c].push_back({chains[c][i].num, i});
            }
        }
        // Sort by uniprot number
        std::sort(sorted_chains[c].begin(), sorted_chains[c].end());
    }
    
    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Try to stitch each sequence from first chain
    for (const auto& [num1, idx1] : sorted_chains[0]) {
        std::vector<size_t> stitch = {idx1};
        std::vector<uint64_t> nums = {num1};
        bool valid = true;
        
        // Find compatible sequences in other chains
        for (size_t c = 1; c < num_chains && valid; c++) {
            std::vector<size_t> compatible;
            
            // Binary search for range of possibly compatible sequences
            uint64_t min_val = (num1 > max_distance) ? num1 - max_distance : 0;
            uint64_t max_val = num1 + max_distance;
            
            auto lower = std::lower_bound(sorted_chains[c].begin(), sorted_chains[c].end(),
                std::make_pair(min_val, size_t(0)));
            auto upper = std::upper_bound(sorted_chains[c].begin(), sorted_chains[c].end(),
                std::make_pair(max_val, SIZE_MAX));
            
            // Check each candidate
            for (auto it = lower; it != upper; ++it) {
                bool is_compatible = true;
                uint64_t num = it->first;
                
                // Check distance to all previous sequences
                for (uint64_t prev_num : nums) {
                    uint64_t dist = (num > prev_num) ? num - prev_num : prev_num - num;
                    if (dist > max_distance) {
                        is_compatible = false;
                        break;
                    }
                }
                
                if (is_compatible) {
                    compatible.push_back(it->second);
                }
            }
            
            if (compatible.empty()) {
                valid = false;
            } else {
                // Randomly select one
                std::uniform_int_distribution<> dis(0, compatible.size() - 1);
                size_t selected = compatible[dis(gen)];
                stitch.push_back(selected);
                nums.push_back(chains[c][selected].num);
            }
        }
        
        if (valid) {
            results.push_back(std::move(stitch));
        }
    }
    
    return results;
}

// Write output
void write_output(const std::string& filename,
                 const std::vector<std::vector<size_t>>& stitches,
                 const std::vector<std::vector<Sequence>>& chains) {
    
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Error: Cannot write to " << filename << std::endl;
        return;
    }
    
    for (const auto& stitch : stitches) {
        out << ">";
        
        // Write header
        if (stitch.size() == 2) {
            const auto& s1 = chains[0][stitch[0]];
            const auto& s2 = chains[1][stitch[1]];
            uint64_t dist = (s1.num > s2.num) ? s1.num - s2.num : s2.num - s1.num;
            out << s1.uid << "_" << s2.uid << "_" << dist;
        } else {
            for (size_t i = 0; i < stitch.size(); i++) {
                if (i > 0) out << "_";
                out << chains[i][stitch[i]].uid;
            }
        }
        out << "\n";
        
        // Write sequences
        for (size_t i = 0; i < stitch.size(); i++) {
            out << chains[i][stitch[i]].seq;
        }
        out << "\n";
    }
}

void print_usage(const char* program) {
    std::cout << "Genomic Stitcher (Optimized)\n\n";
    std::cout << "Usage: " << program << " -i <file1> <file2> [<file3>...] -o <output> [options]\n\n";
    std::cout << "Required:\n";
    std::cout << "  -i <files>   Input a3m files (at least 2)\n";
    std::cout << "  -o <file>    Output stitched a3m file\n\n";
    std::cout << "Options:\n";
    std::cout << "  -d <int>     Maximum genomic distance (default: 20)\n";
    std::cout << "  -qid <int>   Minimum query identity % (default: 15)\n";
    std::cout << "  -cov <int>   Minimum coverage % (default: 75)\n";
    std::cout << "  -h           Show this help\n";
}

int main(int argc, char* argv[]) {
    // Parse arguments
    std::vector<std::string> input_files;
    std::string output_file;
    uint64_t max_distance = 20;
    float min_coverage = 0.75f;
    float min_identity = 0.15f;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-i") {
            while (++i < argc && argv[i][0] != '-') {
                input_files.push_back(argv[i]);
            }
            i--;
        } else if (arg == "-o" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "-d" && i + 1 < argc) {
            max_distance = std::stoull(argv[++i]);
        } else if (arg == "-qid" && i + 1 < argc) {
            min_identity = std::stoi(argv[++i]) / 100.0f;
        } else if (arg == "-cov" && i + 1 < argc) {
            min_coverage = std::stoi(argv[++i]) / 100.0f;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    if (input_files.size() < 2 || output_file.empty()) {
        std::cerr << "Error: Need at least 2 input files and an output file\n";
        print_usage(argv[0]);
        return 1;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Read and process all chains
    std::vector<std::vector<Sequence>> chains;
    
    for (const auto& file : input_files) {
        std::cout << "Processing " << file << "...\n";
        
        auto sequences = read_and_process_a3m(file, min_coverage, min_identity);
        if (sequences.empty()) {
            std::cerr << "Error: No sequences in " << file << std::endl;
            return 1;
        }
        
        std::cout << "  " << sequences.size() << " sequences after filtering\n";
        chains.push_back(std::move(sequences));
    }
    
    // Stitch sequences
    std::cout << "Stitching sequences...\n";
    auto stitched = stitch_sequences(chains, max_distance);
    
    // Write output
    write_output(output_file, stitched, chains);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "Wrote " << stitched.size() << " sequences to " 
              << output_file << " in " << duration << " ms\n";
    
    return 0;
}
