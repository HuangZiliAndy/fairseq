import os
import argparse
import soundfile as sf

parser = argparse.ArgumentParser(description='Prepare dataset')
parser.add_argument('input_dir', type=str, help='Input directory')
parser.add_argument('output_dir', type=str, help='Output directory')
parser.add_argument('cfgname', type=str, help='cfg name')
args = parser.parse_args()

def main():
    os.path.exists("{}/wav.scp".format(args.input_dir)) and os.path.exists("{}/text".format(args.input_dir))
    if not os.path.exists(args.output_dir + '/data'):
        os.makedirs(args.output_dir + '/data')
    if not os.path.exists(args.output_dir + '/label'):
        os.makedirs(args.output_dir + '/label')
    data_file = open("{}/data/{}.tsv".format(args.output_dir, args.cfgname), 'w')
    label_file_word = open("{}/label/{}.wrd".format(args.output_dir, args.cfgname), 'w')
    label_file_ltr = open("{}/label/{}.ltr".format(args.output_dir, args.cfgname), 'w')
    with open("{}/wav.scp".format(args.input_dir), 'r') as fh:
        content_wav_scp = fh.readlines()
    with open("{}/text".format(args.input_dir), 'r') as fh:
        content_text = fh.readlines()
    assert len(content_wav_scp) == len(content_text)
    for i in range(len(content_wav_scp)):
        print("{}/{}".format(i+1, len(content_wav_scp)))
        line_wav_scp = content_wav_scp[i].strip('\n')
        line_text = content_text[i].strip('\n')
        line_wav_scp_split = line_wav_scp.split(None, 1)
        line_text_split = line_text.split(None, 1)
        assert line_wav_scp_split[0] == line_text_split[0]
        uttpath = line_wav_scp_split[1]
        assert os.path.exists(uttpath)
        text = line_text_split[1]
        if i == 0:
            data_file.write("{}\n".format('/'.join(uttpath.split('/')[:-1])))
            
        data_file.write("{}\t{}\n".format(uttpath.split('/')[-1], sf.info(uttpath).frames))
        label_file_word.write("{}\n".format(text))
        label_file_ltr.write("{}\n".format(" ".join(list(text.replace(" ", "|"))) + " |"))
    data_file.close()
    label_file_word.close()
    label_file_ltr.close() 
    return 0

if __name__ == '__main__':
    main()
