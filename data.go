package disco

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
)

// Loads the MovieLens 100K dataset.
func LoadMovieLens() (*Dataset[int, string], error) {
	data := NewDataset[int, string]()

	itemPath, err := downloadFile(
		"ml-100k/u.item",
		"https://files.grouplens.org/datasets/movielens/ml-100k/u.item",
		"553841ebc7de3a0fd0d6b62a204ea30c1e651aacfb2814c7a6584ac52f2c5701",
	)
	if err != nil {
		return data, err
	}

	dataPath, err := downloadFile(
		"ml-100k/u.data",
		"https://files.grouplens.org/datasets/movielens/ml-100k/u.data",
		"06416e597f82b7342361e41163890c81036900f418ad91315590814211dca490",
	)
	if err != nil {
		return data, err
	}

	file, err := os.Open(itemPath)
	if err != nil {
		return data, err
	}

	movies := make(map[string]string, 1682)

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		row0, rest, _ := strings.Cut(scanner.Text(), "|")
		row1, _, _ := strings.Cut(rest, "|")
		movies[row0] = convertToUtf8(row1)
	}

	file, err = os.Open(dataPath)
	if err != nil {
		return data, err
	}

	data.Grow(100000)

	scanner = bufio.NewScanner(file)
	for scanner.Scan() {
		row0, rest, _ := strings.Cut(scanner.Text(), "\t")
		row1, rest, _ := strings.Cut(rest, "\t")
		row2, _, _ := strings.Cut(rest, "\t")

		userId, err := strconv.Atoi(row0)
		if err != nil {
			return data, err
		}

		value, err := strconv.ParseFloat(row2, 64)
		if err != nil {
			return data, err
		}

		data.Push(userId, movies[row1], float32(value))
	}

	return data, nil
}

func downloadFile(filename string, url string, fileHash string) (string, error) {
	home := os.Getenv("HOME")
	if home == "" {
		return "", errors.New("No HOME")
	}

	dest := path.Join(home, ".disco", filename)
	_, err := os.Stat(dest)
	if err == nil {
		return dest, nil
	}

	_, err = os.Stat(filepath.Dir(dest))
	if err != nil {
		err = os.MkdirAll(filepath.Dir(dest), 0755)
		if err != nil {
			return "", err
		}
	}

	fmt.Printf("Downloading data from %s\n", url)
	resp, err := http.Get(url)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	contents, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	b := sha256.Sum256(contents)
	checksum := hex.EncodeToString(b[:])
	if checksum != fileHash {
		return "", fmt.Errorf("Bad checksum: %s", checksum)
	}

	f, err := os.Create(dest)
	if err != nil {
		return "", err
	}
	defer f.Close()

	_, err = f.Write(contents)
	if err != nil {
		return "", err
	}

	return dest, nil
}

func convertToUtf8(str string) string {
	buf := make([]byte, 0, len(str))
	// iterate over bytes
	for i := 0; i < len(str); i++ {
		v := str[i]
		// ISO-8859-1 to UTF-8
		// first 128 are same
		if v < 128 {
			buf = append(buf, v)
		} else {
			buf = append(buf, 195, v-64)
		}
	}
	return string(buf)
}
