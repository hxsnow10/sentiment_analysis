<?php
error_reporting(E_ALL ^ E_NOTICE);
require_once './src/QcloudApi/QcloudApi.php';

$config = array(
                'SecretId'       => 'AKIDkwgRJUxDIDDdg84rhMdkamxd7V7LD1wR',
                'SecretKey'      => 'EQDQ2RWIXwm8EmBSvGvfCFnusuzlqOQv',
                'RequestMethod'  => 'POST',
                'DefaultRegion'  => 'gz');

$wenzhi = QcloudApi::load(QcloudApi::MODULE_WENZHI, $config);

$package = array("content"=>"腾讯入股京东");

$a = $wenzhi->TextClassify($package);

if ($a === false) {
    $error = $wenzhi->getError();
    echo "Error code:" . $error->getCode() . ".\n";
    echo "message:" . $error->getMessage() . ".\n";
    echo "ext:" . var_export($error->getExt(), true) . ".\n";
} else {
    var_dump($a);
}

echo "\nRequest :" . $wenzhi->getLastRequest();
echo "\nResponse :" . $wenzhi->getLastResponse();
echo "\n";

