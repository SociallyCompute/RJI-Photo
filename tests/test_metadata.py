from unittest import TestCase

from preprocessing.metadata import MetaDataReader
import pprint


class TestMetaDataReader(TestCase):
    def test_get_cc_from_file(self):
        test_dir = r'/media/matt/4TBInternal/zzFootball/20160910_MUvsEasternMichigan/For_Upload/'
        test_file = test_dir + '20160910_MUvsEasternMichigan_EC_0001.JPG'

        # First reader should act normally
        reader1 = MetaDataReader(test_dir)
        exif = reader1.get_cc_from_file(test_file)
        self.assertTrue(exif)

        # Verify list of files is returned
        img_list = reader1.get_cc_from_file()
        self.assertTrue(img_list)

        # Catch bad file paths
        self.assertRaises(FileNotFoundError, reader1.get_cc_from_file, 'asbdsjakl;')

        # Second reader should fail on file
        self.assertRaises(NotADirectoryError, MetaDataReader, test_file)

        # Third reader should work unless nothing provided at all
        reader3 = MetaDataReader()
        exif = reader3.get_cc_from_file(test_file)
        self.assertTrue(exif)

        self.assertRaises(NotADirectoryError, reader3.get_cc_from_file)

    def test_get_cc_from_dir(self):
        test_dir = r'/media/matt/4TBInternal/zzFootball/20160910_MUvsEasternMichigan/For_Upload/'
        reader1 = MetaDataReader()
        cc_dir = reader1.get_cc_from_dir(test_dir)
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(cc_dir)
