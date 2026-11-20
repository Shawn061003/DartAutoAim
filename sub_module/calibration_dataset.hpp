#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

namespace calibration_dataset {
namespace fs=std::filesystem;

// 采集追加和图片归档共用锁，避免跨进程分配重名文件或同时移动图片。
class Lock {
    int descriptor_=-1;
public:
    explicit Lock(const fs::path& root) {
        fs::create_directories(root);
        descriptor_=::open((root/".dataset.lock").c_str(),O_CREAT|O_RDWR|O_CLOEXEC,0666);
        if(descriptor_<0) throw std::runtime_error("Cannot open dataset lock: "+root.string());
        if(::flock(descriptor_,LOCK_EX)!=0) {
            ::close(descriptor_);descriptor_=-1;
            throw std::runtime_error("Cannot lock dataset: "+root.string());
        }
    }
    ~Lock() {if(descriptor_>=0) ::close(descriptor_);}
    Lock(const Lock&)=delete;
    Lock& operator=(const Lock&)=delete;
};

inline std::vector<fs::path> images(const fs::path& directory) {
    std::vector<fs::path> paths;
    if(!fs::exists(directory)) return paths;
    for(const auto& entry:fs::directory_iterator(directory)) {
        auto extension=entry.path().extension().string();
        std::transform(extension.begin(),extension.end(),extension.begin(),[](unsigned char c){return std::tolower(c);});
        if(entry.is_regular_file() && extension==".bmp" && entry.path().stem().string().find(".pending")==std::string::npos)
            paths.push_back(fs::absolute(entry.path()));
    }
    return paths;
}

inline std::uint64_t nextNumber(const std::vector<std::string>& keys) {
    std::uint64_t next=1;
    for(const auto& key:keys) {
        if(key.empty() || !std::all_of(key.begin(),key.end(),[](unsigned char c){return std::isdigit(c);})) continue;
        const auto number=std::stoull(key);
        if(number==std::numeric_limits<std::uint64_t>::max()) throw std::runtime_error("Capture filename sequence exhausted");
        next=std::max(next,static_cast<std::uint64_t>(number+1));
    }
    return next;
}

struct Move {fs::path* path;fs::path destination;};

// 注解保存失败时逆序回滚，包括双目中途只移动成功一侧的情况。
template<class Commit> void moveImages(const std::vector<Move>& moves,Commit commit) {
    std::vector<fs::path> originals;
    for(const auto& move:moves) {
        if(fs::exists(move.destination)) throw std::runtime_error("Image already exists: "+move.destination.string());
        fs::create_directories(move.destination.parent_path());
        originals.push_back(*move.path);
    }
    std::size_t completed=0;
    try {
        for(const auto& move:moves) {
            fs::rename(*move.path,move.destination);*move.path=move.destination;++completed;
        }
        commit();
    } catch(...) {
        while(completed>0) {
            --completed;const auto& move=moves[completed];
            fs::rename(*move.path,originals[completed]);*move.path=originals[completed];
        }
        throw;
    }
}
} // namespace calibration_dataset
